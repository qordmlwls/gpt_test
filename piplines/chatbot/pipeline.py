import os
from typing import Dict
from sagemaker.workflow.entities import PipelineVariable

import boto3
import sagemaker
import sagemaker.session
from sagemaker.huggingface import HuggingFace
from sagemaker.inputs import (
    TrainingInput, CreateModelInput
)
from sagemaker.huggingface.model import HuggingFaceModel
from sagemaker.sklearn.processing import SKLearnProcessor
from sagemaker.processing import (
    ProcessingInput, ProcessingOutput, Processor,
)
from sagemaker.workflow.pipeline import Pipeline
from sagemaker.workflow.pipeline_context import PipelineSession
from sagemaker.workflow.steps import (
    ProcessingStep, TrainingStep, CreateModelStep
)

BASE_DIR = os.path.dirname(os.path.realpath(__file__))


def get_sagemaker_client(region):
    """Gets the sagemaker client.
       Args:
           region: the aws region to start the session
       Returns:
           `sagemaker.session.Session instance
       """
    boto_session = boto3.Session(region_name=region)
    sagemaker_client = boto_session.client("sagemaker")
    return sagemaker_client


def get_session(region, default_bucket):
    """Gets the sagemaker session based on the region.
    Args:
        region: the aws region to start the session
        default_bucket: the bucket to use for storing the artifacts
    Returns:
        `sagemaker.session.Session instance
    """

    boto_session = boto3.Session(region_name=region)

    sagemaker_client = boto_session.client("sagemaker")
    runtime_client = boto_session.client("sagemaker-runtime")
    return sagemaker.session.Session(
        boto_session=boto_session,
        sagemaker_client=sagemaker_client,
        sagemaker_runtime_client=runtime_client,
        default_bucket=default_bucket,
    )


def get_pipeline_session(region, default_bucket):
    """Gets the pipeline session based on the region.
    Args:
        region: the aws region to start the session
        default_bucket: the bucket to use for storing the artifacts
    Returns:
        PipelineSession instance
    """

    boto_session = boto3.Session(region_name=region)
    sagemaker_client = boto_session.client("sagemaker")

    return PipelineSession(
        boto_session=boto_session,
        sagemaker_client=sagemaker_client,
        default_bucket=default_bucket,
    )


def get_preprocessing_step(role,
                           image_uri,
                           instance_type,
                           pipeline_session):
    processor = Processor(
        entrypoint=['python3', 'code/pipelines/preprocessing.py'],
        image_uri=image_uri,
        role=role,
        instance_type=instance_type,
        instance_count=1,
        sagemaker_session=pipeline_session
    )
    return ProcessingStep(
        name="Preprocessing",
        processor=processor,
        inputs=[
            ProcessingInput(
                source='s3://sagemaker-sample-data-{}/personalize/movielens/'.format('ap-northeast-2'),
                destination='/opt/ml/processing/input')
        ],
        outputs=[
            ProcessingOutput(output_name="output", source="/opt/ml/processing/output"),
        ]
    )


def get_train_step(role,
                      image_uri,
                      instance_type,
                      pipeline_session,
                      inputs):
    estimator = HuggingFace(
        entry_point='train.py',
        source_dir='code/pipelines/train.py',
        instance_type=instance_type,
        instance_count=1,
        transformers_version='4.6',
        pytorch_version='1.7',
        py_version='py36',
        role=role,
        image_uri=image_uri,
        sagemaker_session=pipeline_session
    )
    return TrainingStep(
        name="Training",
        estimator=estimator,
        inputs=inputs
    )


def get_create_model_step(role,
                          image_uri,
                          endpoint_instance_type,
                          pipeline_session,
                          model_data: PipelineVariable):
    model = HuggingFaceModel(
        image_uri=image_uri,
        role=role,
        model_data=model_data,
        pytorch_version="1.12.1",
        sagemaker_session=pipeline_session
    )
    inputs = CreateModelInput(
        instance_type=endpoint_instance_type
    )
    return CreateModelStep(
        name='CreatingModel',
        model=model,
        inputs=inputs
    )


def get_deploy_processing_step(role,
                               instance_type,
                               pipeline_session,
                               model_name: PipelineVariable,
                               endpoint_instance_type,
                               endpoint_instance_count):
    processor = SKLearnProcessor(
        framework_version='0.23-1',
        role=role,
        instance_type=instance_type,
        instance_count=1,
        sagemaker_session=pipeline_session
    )
    return ProcessingStep(
        name='DeployingModel',
        processor=processor,
        job_arguments=[
            "--model_name", model_name,
            "--endpoint_instance_type", endpoint_instance_type,
            "--endpoint_instance_count", endpoint_instance_count,
            "--endpoint_name", "Chatbot-Endpoint"
        ],
        code="s3://test-sagemaker/deploy_code/deploy_model.py"
    )


def get_pipeline(
        region,
        role=None,
        default_bucket=None,
        pipeline_name="PersonalizeTrainPipeline",
        train_instance_type='ml.p3.8xlarge',
        endpoint_instance_type="ml.m5.2xlarge",
        endpoint_instance_count=2):
    sagemaker_session = get_session(region, default_bucket)
    if role is None:
        role = sagemaker.session.get_execution_role(sagemaker_session)

    processing_image_uri = 'test.dkr.ecr.ap-northeast-2.amazonaws.com/sagemaker-test:latest'
    training_image_uri = 'test.dkr.ecr.ap-northeast-2.amazonaws.com/sagemaker-test:train-step'
    inference_image_uri = 'test.dkr.ecr.ap-northeast-2.amazonaws.com/sagemaker-test:inference-step'

    pipeline_session = get_pipeline_session(region, default_bucket)

    step_preprocess = get_preprocessing_step(role,
                                             processing_image_uri,
                                             pipeline_session=pipeline_session,
                                             instance_type="ml.m5.12xlarge"
                                             )
    training_inputs = {
        "output": TrainingInput(
            s3_data=step_preprocess.properties.ProcessingOutputConfig.Outputs[
                "output"
            ].S3Output.S3Uri
        )
    }
    step_train = get_train_step(role,
                                training_image_uri,
                                instance_type=train_instance_type,
                                pipeline_session=pipeline_session,
                                inputs=training_inputs
                                )
    model_data = step_train.properties.ModelArtifacts.S3ModelArtifacts
    step_create_model = get_create_model_step(role,
                                              inference_image_uri,
                                              endpoint_instance_type=endpoint_instance_type,
                                              pipeline_session=pipeline_session,
                                              model_data=model_data
                                              )
    model_name = step_create_model.properties.ModelName
    step_deploy = get_deploy_processing_step(role,
                                             instance_type="ml.m5.xlarge",
                                             pipeline_session=pipeline_session,
                                             model_name=model_name,
                                             endpoint_instance_type=endpoint_instance_type,
                                             endpoint_instance_count=str(endpoint_instance_count)
                                             )

    pipeline = Pipeline(
        name=pipeline_name,
        parameters=[],
        steps=[step_preprocess, step_train, step_create_model, step_deploy],
        sagemaker_session=pipeline_session,
    )
    return pipeline
