"""Tests for advanced-tier tasks — verifies multi-service, multi-step grading.

Advanced tasks require the agent to execute ordered commands across multiple AWS
services. The grader checks both step completion and service usage via the
EpisodeTracker.

Run inside Docker:
    docker exec aws-rl-env python -m pytest tests/test_advanced_tasks.py -v
"""

import json

import pytest
import yaml
from pathlib import Path

from models import SuccessCriteria, Task, TaskID, TaskDifficulty, SetupCommand
from server.services.aws_backend import AwsBackend
from server.services.task_grader import TaskGrader
from server.services.episode_tracker import EpisodeTracker

TASKS_FILE = (
    Path(__file__).resolve().parent.parent
    / "server"
    / "services"
    / "tasks"
    / "advanced.yaml"
)

_LAMBDA_CODE = "--code S3Bucket=dummy,S3Key=dummy.zip"
_ROLE = "arn:aws:iam::000000000000:role"
_SIMPLE_POLICY = '\'{"Version":"2012-10-17","Statement":[{"Effect":"Allow","Action":"s3:GetObject","Resource":"*"}]}\''


def _run(backend: AwsBackend, cmd: str) -> tuple[str, bool, str, str]:
    """Execute a command and return (cmd, success, stdout, stderr)."""
    success, stdout, stderr = backend.execute_command(cmd)
    return (cmd, success, stdout, stderr)


def _assume(service: str) -> str:
    """Build an assume-role-policy-document JSON for a given AWS service."""
    doc = json.dumps(
        {
            "Version": "2012-10-17",
            "Statement": [
                {
                    "Effect": "Allow",
                    "Principal": {"Service": service},
                    "Action": "sts:AssumeRole",
                }
            ],
        }
    )
    return f"'{doc}'"


def _execute_task(
    task_id: int, backend: AwsBackend
) -> list[tuple[str, bool, str, str]]:
    """Execute the full command sequence for a task, returning all results.

    Handles dynamic ID discovery inline — commands are built and executed
    sequentially, each using outputs from prior commands as needed.
    """
    R: list[tuple[str, bool, str, str]] = []
    run = lambda cmd: R.append(_run(backend, cmd)) or R[-1]  # noqa: E731

    if task_id == 15:
        run(
            f"aws iam create-role --role-name processor-role --assume-role-policy-document {_assume('lambda.amazonaws.com')}"
        )
        run(
            f"aws lambda create-function --function-name processor --runtime python3.12 --handler index.handler --role {_ROLE}/processor-role {_LAMBDA_CODE}"
        )
        run("aws sqs create-queue --queue-name work-items")
        run(
            "aws lambda create-event-source-mapping --function-name processor --event-source-arn arn:aws:sqs:us-east-1:000000000000:work-items --batch-size 10"
        )

    elif task_id == 16:
        run(
            "aws dynamodb create-table --table-name products --key-schema AttributeName=product_id,KeyType=HASH --attribute-definitions AttributeName=product_id,AttributeType=S --billing-mode PAY_PER_REQUEST"
        )
        run(
            f"aws iam create-role --role-name product-api-role --assume-role-policy-document {_assume('lambda.amazonaws.com')}"
        )
        run(
            f"aws lambda create-function --function-name product-api --runtime python3.12 --handler index.handler --role {_ROLE}/product-api-role {_LAMBDA_CODE}"
        )
        _, _, api_out, _ = run("aws apigateway create-rest-api --name products-api")
        api_id = json.loads(api_out)["id"]
        _, _, res_list, _ = run(f"aws apigateway get-resources --rest-api-id {api_id}")
        root_id = next(
            i["id"] for i in json.loads(res_list)["items"] if i["path"] == "/"
        )
        _, _, res_out, _ = run(
            f"aws apigateway create-resource --rest-api-id {api_id} --parent-id {root_id} --path-part products"
        )
        res_id = json.loads(res_out)["id"]
        run(
            f"aws apigateway put-method --rest-api-id {api_id} --resource-id {res_id} --http-method GET --authorization-type NONE"
        )
        run(
            f"aws apigateway put-integration --rest-api-id {api_id} --resource-id {res_id} --http-method GET --type AWS_PROXY --integration-http-method POST --uri arn:aws:apigateway:us-east-1:lambda:path/2015-03-31/functions/arn:aws:lambda:us-east-1:000000000000:function:product-api/invocations"
        )

    elif task_id == 17:
        run("aws sns create-topic --name order-events")
        run("aws sqs create-queue --queue-name shipping-queue")
        run("aws sqs create-queue --queue-name billing-queue")
        run(
            "aws sns subscribe --topic-arn arn:aws:sns:us-east-1:000000000000:order-events --protocol sqs --notification-endpoint arn:aws:sqs:us-east-1:000000000000:shipping-queue"
        )
        run(
            "aws sns subscribe --topic-arn arn:aws:sns:us-east-1:000000000000:order-events --protocol sqs --notification-endpoint arn:aws:sqs:us-east-1:000000000000:billing-queue"
        )
        run(
            'aws sns publish --topic-arn arn:aws:sns:us-east-1:000000000000:order-events --message "test order event"'
        )

    elif task_id == 87:
        run("aws s3api create-bucket --bucket image-uploads")
        run(
            f"aws iam create-role --role-name image-resizer-role --assume-role-policy-document {_assume('lambda.amazonaws.com')}"
        )
        run(
            f"aws lambda create-function --function-name image-resizer --runtime python3.12 --handler index.handler --role {_ROLE}/image-resizer-role {_LAMBDA_CODE}"
        )
        run(
            'aws s3api put-bucket-notification-configuration --bucket image-uploads --notification-configuration \'{"LambdaFunctionConfigurations":[{"LambdaFunctionArn":"arn:aws:lambda:us-east-1:000000000000:function:image-resizer","Events":["s3:ObjectCreated:*"]}]}\''
        )
        run(
            'aws events put-rule --name image-upload-rule --schedule-expression "rate(1 hour)"'
        )
        run(
            "aws events put-targets --rule image-upload-rule --targets Id=1,Arn=arn:aws:lambda:us-east-1:000000000000:function:image-resizer"
        )

    elif task_id == 88:
        run(
            f"aws iam create-role --role-name ecs-exec-role --assume-role-policy-document {_assume('ecs-tasks.amazonaws.com')}"
        )
        run(
            'aws ecs register-task-definition --family web-app-task --container-definitions \'[{"name":"web","image":"nginx","memory":256,"cpu":128}]\' --requires-compatibilities FARGATE --network-mode awsvpc --cpu 256 --memory 512'
        )
        run("aws ecs create-cluster --cluster-name web-cluster")
        _, _, tg_out, _ = run(
            "aws elbv2 create-target-group --name web-tg --protocol HTTP --port 80 --vpc-id vpc-00000001 --target-type ip"
        )
        tg_arn = json.loads(tg_out)["TargetGroups"][0]["TargetGroupArn"]
        _, _, lb_out, _ = run(
            "aws elbv2 create-load-balancer --name web-alb --subnets subnet-00000001 subnet-00000002"
        )
        lb_arn = json.loads(lb_out)["LoadBalancers"][0]["LoadBalancerArn"]
        run(
            'aws ec2 create-security-group --group-name ecs-sg --description "ECS tasks"'
        )
        run(
            f"aws elbv2 create-listener --load-balancer-arn {lb_arn} --protocol HTTP --port 80 --default-actions Type=forward,TargetGroupArn={tg_arn}"
        )
        run(
            f"aws ecs create-service --cluster web-cluster --service-name web-service --task-definition web-app-task --desired-count 1 --launch-type FARGATE --network-configuration awsvpcConfiguration={{subnets=[subnet-00000001],securityGroups=[sg-00000001]}} --load-balancers targetGroupArn={tg_arn},containerName=web,containerPort=80"
        )

    elif task_id == 89:
        run(
            "aws dynamodb create-table --table-name orders --key-schema AttributeName=order_id,KeyType=HASH --attribute-definitions AttributeName=order_id,AttributeType=S --billing-mode PAY_PER_REQUEST"
        )
        run("aws sqs create-queue --queue-name order-queue")
        run("aws sns create-topic --name order-notifications")
        run(
            "aws sns subscribe --topic-arn arn:aws:sns:us-east-1:000000000000:order-notifications --protocol sqs --notification-endpoint arn:aws:sqs:us-east-1:000000000000:order-queue"
        )
        run(
            f"aws iam create-role --role-name order-processor-role --assume-role-policy-document {_assume('lambda.amazonaws.com')}"
        )
        run(
            f"aws lambda create-function --function-name order-processor --runtime python3.12 --handler index.handler --role {_ROLE}/order-processor-role {_LAMBDA_CODE}"
        )
        run(
            "aws lambda create-event-source-mapping --function-name order-processor --event-source-arn arn:aws:sqs:us-east-1:000000000000:order-queue --batch-size 10"
        )

    elif task_id == 90:
        run(
            'aws rds create-db-subnet-group --db-subnet-group-name db-subnets --db-subnet-group-description "DB subnets" --subnet-ids subnet-00000001 subnet-00000002'
        )
        run(
            "aws rds create-db-instance --db-instance-identifier app-db --engine mysql --db-instance-class db.t3.micro --master-username admin --master-user-password Password123"
        )
        run(
            'aws secretsmanager create-secret --name db-credentials --secret-string \'{"username":"admin","password":"Password123"}\''
        )
        run(
            f"aws iam create-role --role-name secret-rotator-role --assume-role-policy-document {_assume('lambda.amazonaws.com')}"
        )
        run(
            f"aws lambda create-function --function-name secret-rotator --runtime python3.12 --handler index.handler --role {_ROLE}/secret-rotator-role {_LAMBDA_CODE}"
        )

    elif task_id == 91:
        run(
            'aws ec2 create-security-group --group-name web-sg --description "HTTP access"'
        )
        _, _, tg_out, _ = run(
            "aws elbv2 create-target-group --name frontend-tg --protocol HTTP --port 80 --vpc-id vpc-00000001 --target-type ip"
        )
        tg_arn = json.loads(tg_out)["TargetGroups"][0]["TargetGroupArn"]
        _, _, lb_out, _ = run(
            "aws elbv2 create-load-balancer --name frontend-alb --subnets subnet-00000001 subnet-00000002"
        )
        lb_arn = json.loads(lb_out)["LoadBalancers"][0]["LoadBalancerArn"]
        run(
            f"aws elbv2 create-listener --load-balancer-arn {lb_arn} --protocol HTTP --port 80 --default-actions Type=forward,TargetGroupArn={tg_arn}"
        )
        _, _, hz_out, _ = run(
            "aws route53 create-hosted-zone --name example.internal --caller-reference ref-91"
        )
        hz_id = json.loads(hz_out)["HostedZone"]["Id"].split("/")[-1]
        batch = json.dumps(
            {
                "Changes": [
                    {
                        "Action": "CREATE",
                        "ResourceRecordSet": {
                            "Name": "example.internal",
                            "Type": "A",
                            "TTL": 300,
                            "ResourceRecords": [{"Value": "1.2.3.4"}],
                        },
                    }
                ]
            }
        )
        run(
            f"aws route53 change-resource-record-sets --hosted-zone-id {hz_id} --change-batch '{batch}'"
        )

    elif task_id == 92:
        _, _, pool_out, _ = run(
            "aws cognito-idp create-user-pool --pool-name app-users"
        )
        pool_id = json.loads(pool_out)["UserPool"]["Id"]
        run(
            f"aws cognito-idp create-user-pool-client --user-pool-id {pool_id} --client-name app-client"
        )
        run(
            f"aws iam create-role --role-name auth-handler-role --assume-role-policy-document {_assume('lambda.amazonaws.com')}"
        )
        run(
            f"aws lambda create-function --function-name auth-handler --runtime python3.12 --handler index.handler --role {_ROLE}/auth-handler-role {_LAMBDA_CODE}"
        )
        _, _, api_out, _ = run(
            "aws apigatewayv2 create-api --name auth-api --protocol-type HTTP"
        )
        api_id = json.loads(api_out)["ApiId"]
        run(
            f"aws apigatewayv2 create-authorizer --api-id {api_id} --authorizer-type JWT --name cognito-auth --identity-source $request.header.Authorization --jwt-configuration Issuer=https://cognito-idp.us-east-1.amazonaws.com/{pool_id},Audience={pool_id}"
        )

    elif task_id == 93:
        run("aws s3api create-bucket --bucket cfn-templates")
        run(
            "aws s3api put-object --bucket cfn-templates --key template.yaml --content-type application/x-yaml"
        )
        run(
            f"aws iam create-role --role-name cfn-deploy-role --assume-role-policy-document {_assume('cloudformation.amazonaws.com')}"
        )
        run(
            'aws cloudformation create-stack --stack-name app-stack --template-body \'{"AWSTemplateFormatVersion":"2010-09-09","Resources":{}}\''
        )

    elif task_id == 94:
        run("aws s3api create-bucket --bucket data-lake-raw")
        run("aws s3api create-bucket --bucket data-lake-processed")
        run(
            f"aws iam create-role --role-name glue-etl-role --assume-role-policy-document {_assume('glue.amazonaws.com')}"
        )
        run('aws glue create-database --database-input \'{"Name":"analytics-db"}\'')
        run(
            f'aws glue create-crawler --name raw-data-crawler --role {_ROLE}/glue-etl-role --database-name analytics-db --targets \'{{"S3Targets":[{{"Path":"s3://data-lake-raw/"}}]}}\''
        )

    elif task_id == 95:
        run("aws s3api create-bucket --bucket event-archive")
        run(
            f"aws iam create-role --role-name firehose-delivery-role --assume-role-policy-document {_assume('firehose.amazonaws.com')}"
        )
        run(
            "aws firehose create-delivery-stream --delivery-stream-name event-stream --s3-destination-configuration RoleARN=arn:aws:iam::000000000000:role/firehose-delivery-role,BucketARN=arn:aws:s3:::event-archive"
        )
        run(
            "aws firehose put-record --delivery-stream-name event-stream --record Data=dGVzdCBldmVudA=="
        )

    elif task_id == 96:
        run(
            f"aws iam create-role --role-name db-cleanup-role --assume-role-policy-document {_assume('lambda.amazonaws.com')}"
        )
        run(
            f"aws lambda create-function --function-name db-cleanup --runtime python3.12 --handler index.handler --role {_ROLE}/db-cleanup-role {_LAMBDA_CODE}"
        )
        run(
            'aws events put-rule --name nightly-cleanup --schedule-expression "cron(0 0 * * ? *)"'
        )
        run(
            "aws events put-targets --rule nightly-cleanup --targets Id=1,Arn=arn:aws:lambda:us-east-1:000000000000:function:db-cleanup"
        )
        run(
            "aws lambda add-permission --function-name db-cleanup --statement-id events-invoke --action lambda:InvokeFunction --principal events.amazonaws.com --source-arn arn:aws:events:us-east-1:000000000000:rule/nightly-cleanup"
        )

    elif task_id == 97:
        run(
            "aws ssm put-parameter --name app-config-db-host --type String --value db.internal.local"
        )
        run(
            "aws ssm put-parameter --name app-config-api-key --type String --value sk-test-123"
        )
        run(
            f"aws iam create-role --role-name config-reader-role --assume-role-policy-document {_assume('lambda.amazonaws.com')}"
        )
        run(
            f"aws lambda create-function --function-name config-reader --runtime python3.12 --handler index.handler --role {_ROLE}/config-reader-role {_LAMBDA_CODE}"
        )
        run(
            'aws events put-rule --name config-refresh --schedule-expression "rate(1 hour)"'
        )
        run(
            "aws events put-targets --rule config-refresh --targets Id=1,Arn=arn:aws:lambda:us-east-1:000000000000:function:config-reader"
        )

    elif task_id == 98:
        _, _, sg_out, _ = run(
            'aws ec2 create-security-group --group-name cache-sg --description "Redis access"'
        )
        sg_id = json.loads(sg_out)["GroupId"]
        run(
            f"aws ec2 authorize-security-group-ingress --group-id {sg_id} --protocol tcp --port 6379 --cidr 10.0.0.0/16"
        )
        run(
            'aws elasticache create-cache-subnet-group --cache-subnet-group-name cache-subnets --cache-subnet-group-description "subnets" --subnet-ids subnet-00000001'
        )
        run(
            f"aws elasticache create-cache-cluster --cache-cluster-id session-store --engine redis --cache-node-type cache.t3.micro --num-cache-nodes 1 --security-group-ids {sg_id}"
        )
        run(
            f"aws iam create-policy --policy-name cache-access --policy-document {_SIMPLE_POLICY}"
        )

    elif task_id == 99:
        _, _, sg_out, _ = run(
            'aws ec2 create-security-group --group-name efs-sg --description "NFS access"'
        )
        sg_id = json.loads(sg_out)["GroupId"]
        run(
            f"aws ec2 authorize-security-group-ingress --group-id {sg_id} --protocol tcp --port 2049 --cidr 10.0.0.0/16"
        )
        _, _, efs_out, _ = run("aws efs create-file-system --creation-token shared-fs")
        fs_id = json.loads(efs_out)["FileSystemId"]
        run(
            f"aws efs create-mount-target --file-system-id {fs_id} --subnet-id subnet-00000001 --security-groups {sg_id}"
        )
        run(
            f"aws iam create-policy --policy-name efs-access --policy-document {_SIMPLE_POLICY}"
        )

    elif task_id == 100:
        run("aws s3api create-bucket --bucket emr-logs")
        run("aws s3api create-bucket --bucket emr-output")
        run(
            f"aws iam create-role --role-name emr-service-role --assume-role-policy-document {_assume('elasticmapreduce.amazonaws.com')}"
        )
        run("aws iam create-instance-profile --instance-profile-name emr-ec2-profile")
        run(
            "aws emr create-cluster --name analytics-cluster --release-label emr-6.15.0 --instance-type m5.xlarge --instance-count 1"
        )

    elif task_id == 101:
        _, _, table_out, _ = run(
            "aws dynamodb create-table --table-name user-activity --key-schema AttributeName=user_id,KeyType=HASH --attribute-definitions AttributeName=user_id,AttributeType=S --billing-mode PAY_PER_REQUEST --stream-specification StreamEnabled=true,StreamViewType=NEW_AND_OLD_IMAGES"
        )
        stream_arn = (
            json.loads(table_out)
            .get("TableDescription", {})
            .get(
                "LatestStreamArn",
                "arn:aws:dynamodb:us-east-1:000000000000:table/user-activity/stream/dummy",
            )
        )
        run("aws sqs create-queue --queue-name activity-dlq")
        run(
            f"aws iam create-role --role-name activity-processor-role --assume-role-policy-document {_assume('lambda.amazonaws.com')}"
        )
        run(
            f"aws lambda create-function --function-name activity-processor --runtime python3.12 --handler index.handler --role {_ROLE}/activity-processor-role {_LAMBDA_CODE}"
        )
        run(
            f"aws lambda create-event-source-mapping --function-name activity-processor --event-source-arn {stream_arn} --starting-position LATEST"
        )

    elif task_id == 102:
        run("aws sns create-topic --name system-alerts")
        run("aws sqs create-queue --queue-name alert-archive")
        run(
            "aws sns subscribe --topic-arn arn:aws:sns:us-east-1:000000000000:system-alerts --protocol sqs --notification-endpoint arn:aws:sqs:us-east-1:000000000000:alert-archive"
        )
        run(
            f"aws iam create-role --role-name alert-handler-role --assume-role-policy-document {_assume('lambda.amazonaws.com')}"
        )
        run(
            f"aws lambda create-function --function-name alert-handler --runtime python3.12 --handler index.handler --role {_ROLE}/alert-handler-role {_LAMBDA_CODE}"
        )
        run(
            "aws sns subscribe --topic-arn arn:aws:sns:us-east-1:000000000000:system-alerts --protocol lambda --notification-endpoint arn:aws:lambda:us-east-1:000000000000:function:alert-handler"
        )
        run(
            'aws sns publish --topic-arn arn:aws:sns:us-east-1:000000000000:system-alerts --message "test alert"'
        )

    elif task_id == 103:
        run(
            "aws dynamodb create-table --table-name tasks-table --key-schema AttributeName=task_id,KeyType=HASH --attribute-definitions AttributeName=task_id,AttributeType=S --billing-mode PAY_PER_REQUEST"
        )
        run(
            f"aws iam create-role --role-name tasks-api-role --assume-role-policy-document {_assume('lambda.amazonaws.com')}"
        )
        run(
            f"aws lambda create-function --function-name tasks-api-handler --runtime python3.12 --handler index.handler --role {_ROLE}/tasks-api-role {_LAMBDA_CODE}"
        )
        _, _, api_out, _ = run(
            "aws apigatewayv2 create-api --name tasks-api --protocol-type HTTP"
        )
        api_id = json.loads(api_out)["ApiId"]
        run(
            f"aws apigatewayv2 create-integration --api-id {api_id} --integration-type AWS_PROXY --integration-uri arn:aws:lambda:us-east-1:000000000000:function:tasks-api-handler --payload-format-version 2.0"
        )
        run(f'aws apigatewayv2 create-route --api-id {api_id} --route-key "GET /tasks"')

    elif task_id == 104:
        run("aws s3api create-bucket --bucket secure-input")
        run("aws s3api create-bucket --bucket secure-output")
        policy = json.dumps(
            {
                "Version": "2012-10-17",
                "Statement": [
                    {
                        "Effect": "Deny",
                        "Principal": "*",
                        "Action": "s3:PutObject",
                        "Resource": "arn:aws:s3:::secure-input/*",
                        "Condition": {
                            "StringNotEquals": {
                                "s3:x-amz-server-side-encryption": "AES256"
                            }
                        },
                    }
                ],
            }
        )
        run(f"aws s3api put-bucket-policy --bucket secure-input --policy '{policy}'")
        run(
            f"aws iam create-role --role-name data-transformer-role --assume-role-policy-document {_assume('lambda.amazonaws.com')}"
        )
        run(
            f"aws lambda create-function --function-name data-transformer --runtime python3.12 --handler index.handler --role {_ROLE}/data-transformer-role {_LAMBDA_CODE}"
        )

    elif task_id == 105:
        run(
            "aws secretsmanager create-secret --name third-party-api-key --secret-string sk-live-abc123"
        )
        run(
            f"aws iam create-role --role-name external-caller-role --assume-role-policy-document {_assume('lambda.amazonaws.com')}"
        )
        run(
            f"aws lambda create-function --function-name external-caller --runtime python3.12 --handler index.handler --role {_ROLE}/external-caller-role {_LAMBDA_CODE}"
        )
        _, _, api_out, _ = run("aws apigateway create-rest-api --name external-api")
        api_id = json.loads(api_out)["id"]
        _, _, res_list, _ = run(f"aws apigateway get-resources --rest-api-id {api_id}")
        root_id = next(
            i["id"] for i in json.loads(res_list)["items"] if i["path"] == "/"
        )
        _, _, res_out, _ = run(
            f"aws apigateway create-resource --rest-api-id {api_id} --parent-id {root_id} --path-part call"
        )
        res_id = json.loads(res_out)["id"]
        run(
            f"aws apigateway put-method --rest-api-id {api_id} --resource-id {res_id} --http-method GET --authorization-type NONE"
        )
        run(
            f"aws apigateway put-integration --rest-api-id {api_id} --resource-id {res_id} --http-method GET --type AWS_PROXY --integration-http-method POST --uri arn:aws:apigateway:us-east-1:lambda:path/2015-03-31/functions/arn:aws:lambda:us-east-1:000000000000:function:external-caller/invocations"
        )

    elif task_id == 106:
        run(
            f"aws iam create-role --role-name batch-task-role --assume-role-policy-document {_assume('ecs-tasks.amazonaws.com')}"
        )
        run("aws ecs create-cluster --cluster-name batch-cluster")
        run(
            'aws ecs register-task-definition --family batch-job --container-definitions \'[{"name":"batch","image":"python:3.12","memory":256,"cpu":128}]\' --requires-compatibilities FARGATE --network-mode awsvpc --cpu 256 --memory 512'
        )
        run(
            'aws ec2 create-security-group --group-name batch-sg --description "Batch SG"'
        )
        run(
            "aws ecs run-task --cluster batch-cluster --task-definition batch-job --launch-type FARGATE --network-configuration awsvpcConfiguration={subnets=[subnet-00000001],securityGroups=[sg-00000001]}"
        )

    elif task_id == 107:
        run("aws s3api create-bucket --bucket query-results")
        run("aws s3api create-bucket --bucket analytics-data")
        run('aws glue create-database --database-input \'{"Name":"web-analytics"}\'')
        run(
            f"aws iam create-policy --policy-name athena-access --policy-document {_SIMPLE_POLICY}"
        )
        run(
            "aws athena create-work-group --name analytics-team --configuration ResultConfiguration={OutputLocation=s3://query-results/}"
        )

    elif task_id == 108:
        run("aws s3api create-bucket --bucket lambda-artifacts")
        run(
            "aws s3api put-object --bucket lambda-artifacts --key function.zip --content-type application/zip"
        )
        run(
            f"aws iam create-role --role-name cfn-lambda-role --assume-role-policy-document {_assume('cloudformation.amazonaws.com')}"
        )
        run(
            f"aws iam create-role --role-name lambda-exec-role --assume-role-policy-document {_assume('lambda.amazonaws.com')}"
        )
        run(
            'aws cloudformation create-stack --stack-name lambda-stack --template-body \'{"AWSTemplateFormatVersion":"2010-09-09","Resources":{}}\''
        )

    return R


# All task IDs from the YAML
ALL_TASK_IDS = [
    15,
    16,
    17,
    87,
    88,
    89,
    90,
    91,
    92,
    93,
    94,
    95,
    96,
    97,
    98,
    99,
    100,
    101,
    102,
    103,
    104,
    105,
    106,
    107,
    108,
]


@pytest.fixture
def backend() -> AwsBackend:
    b = AwsBackend()
    b.reset_environment()
    return b


@pytest.fixture
def grader(backend: AwsBackend) -> TaskGrader:
    return TaskGrader(backend)


@pytest.fixture(scope="module")
def advanced_tasks() -> list[dict]:
    with open(TASKS_FILE) as f:
        return yaml.safe_load(f)


def _build_task(entry: dict) -> Task:
    return Task(
        task_id=TaskID(entry["task_id"]),
        difficulty=TaskDifficulty.ADVANCED,
        description=entry["description"],
        success_criteria=SuccessCriteria(**entry.get("success_criteria", {})),
        setup_commands=[
            SetupCommand(command=cmd) if isinstance(cmd, str) else SetupCommand(**cmd)
            for cmd in entry.get("setup_commands", [])
        ],
    )


def test_all_advanced_tasks_have_commands(advanced_tasks: list[dict]) -> None:
    """Every advanced task in the YAML must have a corresponding test."""
    missing = [t["task_id"] for t in advanced_tasks if t["task_id"] not in ALL_TASK_IDS]
    assert not missing, f"No test commands mapped for task_ids: {missing}"


@pytest.mark.parametrize(
    "task_id", ALL_TASK_IDS, ids=[f"task_{t}" for t in ALL_TASK_IDS]
)
def test_advanced_task_commands_execute(task_id: int, backend: AwsBackend) -> None:
    """All commands must execute successfully against MiniStack."""
    results = _execute_task(task_id, backend)
    for i, (cmd, success, stdout, stderr) in enumerate(results):
        assert success, (
            f"Command {i + 1}/{len(results)} failed for task {task_id}.\n"
            f"  Command: {cmd}\n"
            f"  Stderr: {stderr}"
        )


@pytest.mark.parametrize(
    "task_id", ALL_TASK_IDS, ids=[f"task_{t}" for t in ALL_TASK_IDS]
)
def test_advanced_task_grading(
    task_id: int,
    advanced_tasks: list[dict],
    backend: AwsBackend,
    grader: TaskGrader,
) -> None:
    """Execute full sequence and verify grader marks task as achieved."""
    entry = next((t for t in advanced_tasks if t["task_id"] == task_id), None)
    assert entry is not None, f"task_id {task_id} not found in advanced.yaml"

    task = _build_task(entry)
    results = _execute_task(task_id, backend)

    tracker = EpisodeTracker()
    for cmd, success, stdout, stderr in results:
        step = tracker.record_step(cmd, success, stdout, stderr)

    result = grader.grade(task, tracker, step)

    all_cmds = [r[0] for r in results]
    assert result.task_achieved, (
        f"Task {task_id} not achieved.\n"
        f"  Description: {entry['description']}\n"
        f"  Commands: {all_cmds}\n"
        f"  Reason: {result.reason}\n"
        f"  Reward: {result.reward}"
    )
    assert result.reward == 1.0, f"Expected reward=1.0, got {result.reward}"


@pytest.mark.parametrize(
    "task_id", ALL_TASK_IDS, ids=[f"task_{t}_partial" for t in ALL_TASK_IDS]
)
def test_advanced_task_partial_gives_no_completion(
    task_id: int,
    advanced_tasks: list[dict],
    backend: AwsBackend,
    grader: TaskGrader,
) -> None:
    """Executing only the first command should not achieve a multi-step task."""
    entry = next((t for t in advanced_tasks if t["task_id"] == task_id), None)
    assert entry is not None

    steps = entry.get("success_criteria", {}).get("steps", [])
    if len(steps) < 2:
        pytest.skip("Single-step task")

    task = _build_task(entry)

    # Run only the first command
    results = _execute_task(task_id, backend)
    cmd, success, stdout, stderr = results[0]
    tracker = EpisodeTracker()
    step = tracker.record_step(cmd, success, stdout, stderr)
    result = grader.grade(task, tracker, step)

    assert not result.task_achieved, (
        f"Task {task_id} should NOT be achieved with only the first command.\n"
        f"  Command: {cmd}\n  Reason: {result.reason}"
    )
    assert result.reward < 1.0
