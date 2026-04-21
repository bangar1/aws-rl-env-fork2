"""Provides step-by-step solution commands for tasks.

Returns the next command to execute based on how many steps have been completed.
Dynamic IDs are resolved from actual MiniStack state via the backend.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from server.services.environment_strategy import EnvironmentStrategy
    from server.services.episode_tracker import EpisodeTracker

_ROLE = "arn:aws:iam::000000000000:role"
_CODE = "--code S3Bucket=dummy,S3Key=dummy.zip"
_SIMPLE_POLICY = """'{"Version":"2012-10-17","Statement":[{"Effect":"Allow","Action":"s3:GetObject","Resource":"*"}]}'"""


def _assume(svc: str) -> str:
    doc = json.dumps(
        {
            "Version": "2012-10-17",
            "Statement": [
                {
                    "Effect": "Allow",
                    "Principal": {"Service": svc},
                    "Action": "sts:AssumeRole",
                }
            ],
        }
    )
    return f"'{doc}'"


# ---------------------------------------------------------------------------
# Static command sequences (loaded once from test files)
# ---------------------------------------------------------------------------
_static_cache: dict[int, list[str]] | None = None


def _load_static() -> dict[int, list[str]]:
    global _static_cache
    if _static_cache is not None:
        return _static_cache

    import importlib.util

    solutions: dict[int, list[str]] = {}
    tests_dir = Path(__file__).resolve().parent.parent.parent / "tests_tasks"

    for fname, var in [
        ("test_warmup_tasks.py", "WARMUP_COMMANDS"),
        ("test_beginner_tasks.py", "BEGINNER_COMMANDS"),
        ("test_intermediate_tasks.py", "INTERMEDIATE_COMMANDS"),
        ("test_expert_tasks.py", "EXPERT_COMMANDS"),
    ]:
        fpath = tests_dir / fname
        if not fpath.exists():
            continue
        spec = importlib.util.spec_from_file_location(fname.replace(".py", ""), fpath)
        if spec is None or spec.loader is None:
            continue
        mod = importlib.util.module_from_spec(spec)
        try:
            spec.loader.exec_module(mod)
        except Exception:
            continue
        cmds_map = getattr(mod, var, {})
        for tid, cmd in cmds_map.items():
            if isinstance(cmd, str):
                solutions[tid] = [cmd]
            elif isinstance(cmd, list):
                solutions[tid] = [c for c in cmd if isinstance(c, str)]

    _static_cache = solutions
    return solutions


# ---------------------------------------------------------------------------
# Advanced tasks — full command sequences with dynamic ID resolution
# ---------------------------------------------------------------------------


def _advanced_commands(
    task_id: int, backend: EnvironmentStrategy, step: int
) -> list[str]:
    """Return the full ordered command list for an advanced task.

    Some commands depend on outputs from prior steps. We execute discovery
    commands against MiniStack to resolve dynamic IDs.
    """
    a = _assume
    if task_id == 15:
        return [
            f"aws iam create-role --role-name processor-role --assume-role-policy-document {a('lambda.amazonaws.com')}",
            f"aws lambda create-function --function-name processor --runtime python3.12 --handler index.handler --role {_ROLE}/processor-role {_CODE}",
            "aws sqs create-queue --queue-name work-items",
            "aws lambda create-event-source-mapping --function-name processor --event-source-arn arn:aws:sqs:us-east-1:000000000000:work-items --batch-size 10",
        ]

    if task_id == 16:
        cmds = [
            "aws dynamodb create-table --table-name products --key-schema AttributeName=product_id,KeyType=HASH --attribute-definitions AttributeName=product_id,AttributeType=S --billing-mode PAY_PER_REQUEST",
            f"aws iam create-role --role-name product-api-role --assume-role-policy-document {a('lambda.amazonaws.com')}",
            f"aws lambda create-function --function-name product-api --runtime python3.12 --handler index.handler --role {_ROLE}/product-api-role {_CODE}",
            "aws apigateway create-rest-api --name products-api",
        ]
        # Steps 5+ need dynamic IDs — resolve from MiniStack
        if step >= 4:
            ok, out, _ = backend.execute_command("aws apigateway get-rest-apis")
            api_id = "UNKNOWN"
            try:
                for item in json.loads(out).get("items", []):
                    if item.get("name") == "products-api":
                        api_id = item["id"]
                        break
            except Exception:
                pass
            cmds.append(f"aws apigateway get-resources --rest-api-id {api_id}")

            if step >= 5:
                ok2, out2, _ = backend.execute_command(
                    f"aws apigateway get-resources --rest-api-id {api_id}"
                )
                root_id = "UNKNOWN"
                try:
                    for item in json.loads(out2).get("items", []):
                        if item.get("path") == "/":
                            root_id = item["id"]
                            break
                except Exception:
                    pass
                cmds.append(
                    f"aws apigateway create-resource --rest-api-id {api_id} --parent-id {root_id} --path-part products"
                )

            if step >= 6:
                ok3, out3, _ = backend.execute_command(
                    f"aws apigateway get-resources --rest-api-id {api_id}"
                )
                res_id = "UNKNOWN"
                try:
                    for item in json.loads(out3).get("items", []):
                        if item.get("pathPart") == "products":
                            res_id = item["id"]
                            break
                except Exception:
                    pass
                cmds.append(
                    f"aws apigateway put-method --rest-api-id {api_id} --resource-id {res_id} --http-method GET --authorization-type NONE"
                )

            if step >= 7:
                cmds.append(
                    f"aws apigateway put-integration --rest-api-id {api_id} --resource-id {res_id} --http-method GET --type AWS_PROXY --integration-http-method POST --uri arn:aws:apigateway:us-east-1:lambda:path/2015-03-31/functions/arn:aws:lambda:us-east-1:000000000000:function:product-api/invocations"
                )
        return cmds

    if task_id == 17:
        return [
            "aws sns create-topic --name order-events",
            "aws sqs create-queue --queue-name shipping-queue",
            "aws sqs create-queue --queue-name billing-queue",
            "aws sns subscribe --topic-arn arn:aws:sns:us-east-1:000000000000:order-events --protocol sqs --notification-endpoint arn:aws:sqs:us-east-1:000000000000:shipping-queue",
            "aws sns subscribe --topic-arn arn:aws:sns:us-east-1:000000000000:order-events --protocol sqs --notification-endpoint arn:aws:sqs:us-east-1:000000000000:billing-queue",
            'aws sns publish --topic-arn arn:aws:sns:us-east-1:000000000000:order-events --message "test order event"',
        ]

    if task_id == 87:
        return [
            "aws s3api create-bucket --bucket image-uploads",
            f"aws iam create-role --role-name image-resizer-role --assume-role-policy-document {a('lambda.amazonaws.com')}",
            f"aws lambda create-function --function-name image-resizer --runtime python3.12 --handler index.handler --role {_ROLE}/image-resizer-role {_CODE}",
            """aws s3api put-bucket-notification-configuration --bucket image-uploads --notification-configuration '{"LambdaFunctionConfigurations":[{"LambdaFunctionArn":"arn:aws:lambda:us-east-1:000000000000:function:image-resizer","Events":["s3:ObjectCreated:*"]}]}'""",
            'aws events put-rule --name image-upload-rule --schedule-expression "rate(1 hour)"',
            "aws events put-targets --rule image-upload-rule --targets Id=1,Arn=arn:aws:lambda:us-east-1:000000000000:function:image-resizer",
        ]

    if task_id == 88:
        cmds = [
            f"aws iam create-role --role-name ecs-exec-role --assume-role-policy-document {a('ecs-tasks.amazonaws.com')}",
            """aws ecs register-task-definition --family web-app-task --container-definitions '[{"name":"web","image":"nginx","memory":256,"cpu":128}]' --requires-compatibilities FARGATE --network-mode awsvpc --cpu 256 --memory 512""",
            "aws ecs create-cluster --cluster-name web-cluster",
            "aws elbv2 create-target-group --name web-tg --protocol HTTP --port 80 --vpc-id vpc-00000001 --target-type ip",
            "aws elbv2 create-load-balancer --name web-alb --subnets subnet-00000001 subnet-00000002",
            'aws ec2 create-security-group --group-name ecs-sg --description "ECS tasks"',
        ]
        if step >= 6:
            tg_arn = lb_arn = "UNKNOWN"
            ok, out, _ = backend.execute_command(
                "aws elbv2 describe-target-groups --names web-tg"
            )
            try:
                tg_arn = json.loads(out)["TargetGroups"][0]["TargetGroupArn"]
            except Exception:
                pass
            ok, out, _ = backend.execute_command(
                "aws elbv2 describe-load-balancers --names web-alb"
            )
            try:
                lb_arn = json.loads(out)["LoadBalancers"][0]["LoadBalancerArn"]
            except Exception:
                pass
            cmds.append(
                f"aws elbv2 create-listener --load-balancer-arn {lb_arn} --protocol HTTP --port 80 --default-actions Type=forward,TargetGroupArn={tg_arn}"
            )
            if step >= 7:
                cmds.append(
                    f"aws ecs create-service --cluster web-cluster --service-name web-service --task-definition web-app-task --desired-count 1 --launch-type FARGATE --network-configuration awsvpcConfiguration={{subnets=[subnet-00000001],securityGroups=[sg-00000001]}} --load-balancers targetGroupArn={tg_arn},containerName=web,containerPort=80"
                )
        return cmds

    if task_id == 89:
        return [
            "aws dynamodb create-table --table-name orders --key-schema AttributeName=order_id,KeyType=HASH --attribute-definitions AttributeName=order_id,AttributeType=S --billing-mode PAY_PER_REQUEST",
            "aws sqs create-queue --queue-name order-queue",
            "aws sns create-topic --name order-notifications",
            "aws sns subscribe --topic-arn arn:aws:sns:us-east-1:000000000000:order-notifications --protocol sqs --notification-endpoint arn:aws:sqs:us-east-1:000000000000:order-queue",
            f"aws iam create-role --role-name order-processor-role --assume-role-policy-document {a('lambda.amazonaws.com')}",
            f"aws lambda create-function --function-name order-processor --runtime python3.12 --handler index.handler --role {_ROLE}/order-processor-role {_CODE}",
            "aws lambda create-event-source-mapping --function-name order-processor --event-source-arn arn:aws:sqs:us-east-1:000000000000:order-queue --batch-size 10",
        ]

    if task_id == 90:
        return [
            'aws rds create-db-subnet-group --db-subnet-group-name db-subnets --db-subnet-group-description "DB subnets" --subnet-ids subnet-00000001 subnet-00000002',
            "aws rds create-db-instance --db-instance-identifier app-db --engine mysql --db-instance-class db.t3.micro --master-username admin --master-user-password Password123",
            """aws secretsmanager create-secret --name db-credentials --secret-string '{"username":"admin","password":"Password123"}'""",
            f"aws iam create-role --role-name secret-rotator-role --assume-role-policy-document {a('lambda.amazonaws.com')}",
            f"aws lambda create-function --function-name secret-rotator --runtime python3.12 --handler index.handler --role {_ROLE}/secret-rotator-role {_CODE}",
        ]

    if task_id == 91:
        cmds = [
            'aws ec2 create-security-group --group-name web-sg --description "HTTP access"',
            "aws elbv2 create-target-group --name frontend-tg --protocol HTTP --port 80 --vpc-id vpc-00000001 --target-type ip",
            "aws elbv2 create-load-balancer --name frontend-alb --subnets subnet-00000001 subnet-00000002",
        ]
        if step >= 3:
            tg_arn = lb_arn = "UNKNOWN"
            ok, out, _ = backend.execute_command(
                "aws elbv2 describe-target-groups --names frontend-tg"
            )
            try:
                tg_arn = json.loads(out)["TargetGroups"][0]["TargetGroupArn"]
            except Exception:
                pass
            ok, out, _ = backend.execute_command(
                "aws elbv2 describe-load-balancers --names frontend-alb"
            )
            try:
                lb_arn = json.loads(out)["LoadBalancers"][0]["LoadBalancerArn"]
            except Exception:
                pass
            cmds.append(
                f"aws elbv2 create-listener --load-balancer-arn {lb_arn} --protocol HTTP --port 80 --default-actions Type=forward,TargetGroupArn={tg_arn}"
            )
        if step >= 4:
            cmds.append(
                "aws route53 create-hosted-zone --name example.internal --caller-reference ref-91"
            )
        if step >= 5:
            hz_id = "UNKNOWN"
            ok, out, _ = backend.execute_command("aws route53 list-hosted-zones")
            try:
                for hz in json.loads(out).get("HostedZones", []):
                    if "example.internal" in hz.get("Name", ""):
                        hz_id = hz["Id"].split("/")[-1]
                        break
            except Exception:
                pass
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
            cmds.append(
                f"aws route53 change-resource-record-sets --hosted-zone-id {hz_id} --change-batch '{batch}'"
            )
        return cmds

    if task_id == 92:
        cmds = ["aws cognito-idp create-user-pool --pool-name app-users"]
        if step >= 1:
            pool_id = "UNKNOWN"
            ok, out, _ = backend.execute_command(
                "aws cognito-idp list-user-pools --max-results 10"
            )
            try:
                for p in json.loads(out).get("UserPools", []):
                    if "app-users" in p.get("Name", ""):
                        pool_id = p["Id"]
                        break
            except Exception:
                pass
            cmds.append(
                f"aws cognito-idp create-user-pool-client --user-pool-id {pool_id} --client-name app-client"
            )
            cmds.append(
                f"aws iam create-role --role-name auth-handler-role --assume-role-policy-document {a('lambda.amazonaws.com')}"
            )
            cmds.append(
                f"aws lambda create-function --function-name auth-handler --runtime python3.12 --handler index.handler --role {_ROLE}/auth-handler-role {_CODE}"
            )
            cmds.append(
                "aws apigatewayv2 create-api --name auth-api --protocol-type HTTP"
            )
        if step >= 5:
            api_id = "UNKNOWN"
            ok, out, _ = backend.execute_command("aws apigatewayv2 get-apis")
            try:
                for item in json.loads(out).get("Items", []):
                    if item.get("Name") == "auth-api":
                        api_id = item["ApiId"]
                        break
            except Exception:
                pass
            cmds.append(
                f"aws apigatewayv2 create-authorizer --api-id {api_id} --authorizer-type JWT --name cognito-auth --identity-source $request.header.Authorization --jwt-configuration Issuer=https://cognito-idp.us-east-1.amazonaws.com/{pool_id},Audience={pool_id}"
            )
        return cmds

    if task_id == 93:
        return [
            "aws s3api create-bucket --bucket cfn-templates",
            "aws s3api put-object --bucket cfn-templates --key template.yaml --content-type application/x-yaml",
            f"aws iam create-role --role-name cfn-deploy-role --assume-role-policy-document {a('cloudformation.amazonaws.com')}",
            """aws cloudformation create-stack --stack-name app-stack --template-body '{"AWSTemplateFormatVersion":"2010-09-09","Resources":{}}'""",
        ]

    if task_id == 94:
        return [
            "aws s3api create-bucket --bucket data-lake-raw",
            "aws s3api create-bucket --bucket data-lake-processed",
            f"aws iam create-role --role-name glue-etl-role --assume-role-policy-document {a('glue.amazonaws.com')}",
            """aws glue create-database --database-input '{"Name":"analytics-db"}'""",
            f"""aws glue create-crawler --name raw-data-crawler --role {_ROLE}/glue-etl-role --database-name analytics-db --targets '{{"S3Targets":[{{"Path":"s3://data-lake-raw/"}}]}}'""",
        ]

    if task_id == 95:
        return [
            "aws s3api create-bucket --bucket event-archive",
            f"aws iam create-role --role-name firehose-delivery-role --assume-role-policy-document {a('firehose.amazonaws.com')}",
            "aws firehose create-delivery-stream --delivery-stream-name event-stream --s3-destination-configuration RoleARN=arn:aws:iam::000000000000:role/firehose-delivery-role,BucketARN=arn:aws:s3:::event-archive",
            "aws firehose put-record --delivery-stream-name event-stream --record Data=dGVzdCBldmVudA==",
        ]

    if task_id == 96:
        return [
            f"aws iam create-role --role-name db-cleanup-role --assume-role-policy-document {a('lambda.amazonaws.com')}",
            f"aws lambda create-function --function-name db-cleanup --runtime python3.12 --handler index.handler --role {_ROLE}/db-cleanup-role {_CODE}",
            'aws events put-rule --name nightly-cleanup --schedule-expression "cron(0 0 * * ? *)"',
            "aws events put-targets --rule nightly-cleanup --targets Id=1,Arn=arn:aws:lambda:us-east-1:000000000000:function:db-cleanup",
            "aws lambda add-permission --function-name db-cleanup --statement-id events-invoke --action lambda:InvokeFunction --principal events.amazonaws.com --source-arn arn:aws:events:us-east-1:000000000000:rule/nightly-cleanup",
        ]

    if task_id == 97:
        return [
            "aws ssm put-parameter --name app-config-db-host --type String --value db.internal.local",
            "aws ssm put-parameter --name app-config-api-key --type String --value sk-test-123",
            f"aws iam create-role --role-name config-reader-role --assume-role-policy-document {a('lambda.amazonaws.com')}",
            f"aws lambda create-function --function-name config-reader --runtime python3.12 --handler index.handler --role {_ROLE}/config-reader-role {_CODE}",
            'aws events put-rule --name config-refresh --schedule-expression "rate(1 hour)"',
            "aws events put-targets --rule config-refresh --targets Id=1,Arn=arn:aws:lambda:us-east-1:000000000000:function:config-reader",
        ]

    if task_id == 98:
        cmds = [
            'aws ec2 create-security-group --group-name cache-sg --description "Redis access"'
        ]
        if step >= 1:
            sg_id = "UNKNOWN"
            ok, out, _ = backend.execute_command(
                "aws ec2 describe-security-groups --group-names cache-sg"
            )
            try:
                sg_id = json.loads(out)["SecurityGroups"][0]["GroupId"]
            except Exception:
                pass
            cmds.append(
                f"aws ec2 authorize-security-group-ingress --group-id {sg_id} --protocol tcp --port 6379 --cidr 10.0.0.0/16"
            )
            cmds.append(
                'aws elasticache create-cache-subnet-group --cache-subnet-group-name cache-subnets --cache-subnet-group-description "subnets" --subnet-ids subnet-00000001'
            )
            cmds.append(
                f"aws elasticache create-cache-cluster --cache-cluster-id session-store --engine redis --cache-node-type cache.t3.micro --num-cache-nodes 1 --security-group-ids {sg_id}"
            )
            cmds.append(
                f"aws iam create-policy --policy-name cache-access --policy-document {_SIMPLE_POLICY}"
            )
        return cmds

    if task_id == 99:
        cmds = [
            'aws ec2 create-security-group --group-name efs-sg --description "NFS access"'
        ]
        if step >= 1:
            sg_id = "UNKNOWN"
            ok, out, _ = backend.execute_command(
                "aws ec2 describe-security-groups --group-names efs-sg"
            )
            try:
                sg_id = json.loads(out)["SecurityGroups"][0]["GroupId"]
            except Exception:
                pass
            cmds.append(
                f"aws ec2 authorize-security-group-ingress --group-id {sg_id} --protocol tcp --port 2049 --cidr 10.0.0.0/16"
            )
            cmds.append("aws efs create-file-system --creation-token shared-fs")
        if step >= 3:
            fs_id = "UNKNOWN"
            ok, out, _ = backend.execute_command("aws efs describe-file-systems")
            try:
                for fs in json.loads(out).get("FileSystems", []):
                    if fs.get("CreationToken") == "shared-fs":
                        fs_id = fs["FileSystemId"]
                        break
            except Exception:
                pass
            cmds.append(
                f"aws efs create-mount-target --file-system-id {fs_id} --subnet-id subnet-00000001 --security-groups {sg_id}"
            )
            cmds.append(
                f"aws iam create-policy --policy-name efs-access --policy-document {_SIMPLE_POLICY}"
            )
        return cmds

    if task_id == 100:
        return [
            "aws s3api create-bucket --bucket emr-logs",
            "aws s3api create-bucket --bucket emr-output",
            f"aws iam create-role --role-name emr-service-role --assume-role-policy-document {a('elasticmapreduce.amazonaws.com')}",
            "aws iam create-instance-profile --instance-profile-name emr-ec2-profile",
            "aws emr create-cluster --name analytics-cluster --release-label emr-6.15.0 --instance-type m5.xlarge --instance-count 1",
        ]

    if task_id == 101:
        cmds = [
            "aws dynamodb create-table --table-name user-activity --key-schema AttributeName=user_id,KeyType=HASH --attribute-definitions AttributeName=user_id,AttributeType=S --billing-mode PAY_PER_REQUEST --stream-specification StreamEnabled=true,StreamViewType=NEW_AND_OLD_IMAGES",
            "aws sqs create-queue --queue-name activity-dlq",
            f"aws iam create-role --role-name activity-processor-role --assume-role-policy-document {a('lambda.amazonaws.com')}",
            f"aws lambda create-function --function-name activity-processor --runtime python3.12 --handler index.handler --role {_ROLE}/activity-processor-role {_CODE}",
        ]
        if step >= 4:
            stream_arn = "UNKNOWN"
            ok, out, _ = backend.execute_command(
                "aws dynamodb describe-table --table-name user-activity"
            )
            try:
                stream_arn = json.loads(out)["Table"]["LatestStreamArn"]
            except Exception:
                pass
            cmds.append(
                f"aws lambda create-event-source-mapping --function-name activity-processor --event-source-arn {stream_arn} --starting-position LATEST"
            )
        return cmds

    if task_id == 102:
        return [
            "aws sns create-topic --name system-alerts",
            "aws sqs create-queue --queue-name alert-archive",
            "aws sns subscribe --topic-arn arn:aws:sns:us-east-1:000000000000:system-alerts --protocol sqs --notification-endpoint arn:aws:sqs:us-east-1:000000000000:alert-archive",
            f"aws iam create-role --role-name alert-handler-role --assume-role-policy-document {a('lambda.amazonaws.com')}",
            f"aws lambda create-function --function-name alert-handler --runtime python3.12 --handler index.handler --role {_ROLE}/alert-handler-role {_CODE}",
            "aws sns subscribe --topic-arn arn:aws:sns:us-east-1:000000000000:system-alerts --protocol lambda --notification-endpoint arn:aws:lambda:us-east-1:000000000000:function:alert-handler",
            'aws sns publish --topic-arn arn:aws:sns:us-east-1:000000000000:system-alerts --message "test alert"',
        ]

    if task_id == 103:
        cmds = [
            "aws dynamodb create-table --table-name tasks-table --key-schema AttributeName=task_id,KeyType=HASH --attribute-definitions AttributeName=task_id,AttributeType=S --billing-mode PAY_PER_REQUEST",
            f"aws iam create-role --role-name tasks-api-role --assume-role-policy-document {a('lambda.amazonaws.com')}",
            f"aws lambda create-function --function-name tasks-api-handler --runtime python3.12 --handler index.handler --role {_ROLE}/tasks-api-role {_CODE}",
            "aws apigatewayv2 create-api --name tasks-api --protocol-type HTTP",
        ]
        if step >= 4:
            api_id = "UNKNOWN"
            ok, out, _ = backend.execute_command("aws apigatewayv2 get-apis")
            try:
                for item in json.loads(out).get("Items", []):
                    if item.get("Name") == "tasks-api":
                        api_id = item["ApiId"]
                        break
            except Exception:
                pass
            cmds.append(
                f"aws apigatewayv2 create-integration --api-id {api_id} --integration-type AWS_PROXY --integration-uri arn:aws:lambda:us-east-1:000000000000:function:tasks-api-handler --payload-format-version 2.0"
            )
            cmds.append(
                f'aws apigatewayv2 create-route --api-id {api_id} --route-key "GET /tasks"'
            )
        return cmds

    if task_id == 104:
        _spolicy = json.dumps(
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
        return [
            "aws s3api create-bucket --bucket secure-input",
            "aws s3api create-bucket --bucket secure-output",
            f"aws s3api put-bucket-policy --bucket secure-input --policy '{_spolicy}'",
            f"aws iam create-role --role-name data-transformer-role --assume-role-policy-document {a('lambda.amazonaws.com')}",
            f"aws lambda create-function --function-name data-transformer --runtime python3.12 --handler index.handler --role {_ROLE}/data-transformer-role {_CODE}",
        ]

    if task_id == 105:
        cmds = [
            "aws secretsmanager create-secret --name third-party-api-key --secret-string sk-live-abc123",
            f"aws iam create-role --role-name external-caller-role --assume-role-policy-document {a('lambda.amazonaws.com')}",
            f"aws lambda create-function --function-name external-caller --runtime python3.12 --handler index.handler --role {_ROLE}/external-caller-role {_CODE}",
            "aws apigateway create-rest-api --name external-api",
        ]
        if step >= 4:
            api_id = "UNKNOWN"
            ok, out, _ = backend.execute_command("aws apigateway get-rest-apis")
            try:
                for item in json.loads(out).get("items", []):
                    if item.get("name") == "external-api":
                        api_id = item["id"]
                        break
            except Exception:
                pass
            ok2, out2, _ = backend.execute_command(
                f"aws apigateway get-resources --rest-api-id {api_id}"
            )
            root_id = "UNKNOWN"
            try:
                for item in json.loads(out2).get("items", []):
                    if item.get("path") == "/":
                        root_id = item["id"]
                        break
            except Exception:
                pass
            cmds.append(
                f"aws apigateway create-resource --rest-api-id {api_id} --parent-id {root_id} --path-part call"
            )
        if step >= 5:
            res_id = "UNKNOWN"
            ok3, out3, _ = backend.execute_command(
                f"aws apigateway get-resources --rest-api-id {api_id}"
            )
            try:
                for item in json.loads(out3).get("items", []):
                    if item.get("pathPart") == "call":
                        res_id = item["id"]
                        break
            except Exception:
                pass
            cmds.append(
                f"aws apigateway put-method --rest-api-id {api_id} --resource-id {res_id} --http-method GET --authorization-type NONE"
            )
            cmds.append(
                f"aws apigateway put-integration --rest-api-id {api_id} --resource-id {res_id} --http-method GET --type AWS_PROXY --integration-http-method POST --uri arn:aws:apigateway:us-east-1:lambda:path/2015-03-31/functions/arn:aws:lambda:us-east-1:000000000000:function:external-caller/invocations"
            )
        return cmds

    if task_id == 106:
        return [
            f"aws iam create-role --role-name batch-task-role --assume-role-policy-document {a('ecs-tasks.amazonaws.com')}",
            "aws ecs create-cluster --cluster-name batch-cluster",
            """aws ecs register-task-definition --family batch-job --container-definitions '[{"name":"batch","image":"python:3.12","memory":256,"cpu":128}]' --requires-compatibilities FARGATE --network-mode awsvpc --cpu 256 --memory 512""",
            'aws ec2 create-security-group --group-name batch-sg --description "Batch SG"',
            "aws ecs run-task --cluster batch-cluster --task-definition batch-job --launch-type FARGATE --network-configuration awsvpcConfiguration={subnets=[subnet-00000001],securityGroups=[sg-00000001]}",
        ]

    if task_id == 107:
        return [
            "aws s3api create-bucket --bucket query-results",
            "aws s3api create-bucket --bucket analytics-data",
            """aws glue create-database --database-input '{"Name":"web-analytics"}'""",
            f"aws iam create-policy --policy-name athena-access --policy-document {_SIMPLE_POLICY}",
            "aws athena create-work-group --name analytics-team --configuration ResultConfiguration={OutputLocation=s3://query-results/}",
        ]

    if task_id == 108:
        return [
            "aws s3api create-bucket --bucket lambda-artifacts",
            "aws s3api put-object --bucket lambda-artifacts --key function.zip --content-type application/zip",
            f"aws iam create-role --role-name cfn-lambda-role --assume-role-policy-document {a('cloudformation.amazonaws.com')}",
            f"aws iam create-role --role-name lambda-exec-role --assume-role-policy-document {a('lambda.amazonaws.com')}",
            """aws cloudformation create-stack --stack-name lambda-stack --template-body '{"AWSTemplateFormatVersion":"2010-09-09","Resources":{}}'""",
        ]

    return []


# ---------------------------------------------------------------------------
# Expert tasks with dynamic IDs
# ---------------------------------------------------------------------------


def _expert_dynamic_command(
    task_id: int, backend: EnvironmentStrategy, step: int, static_cmds: list[str]
) -> list[str]:
    """Append dynamically resolved commands for expert tasks that need runtime IDs."""
    cmds = list(static_cmds)

    if task_id == 114:
        # Route53 zone-id from setup
        ok, out, _ = backend.execute_command("aws route53 list-hosted-zones")
        zone_id = "UNKNOWN"
        try:
            for hz in json.loads(out).get("HostedZones", []):
                if "example.com" in hz.get("Name", ""):
                    zone_id = hz["Id"].split("/")[-1]
                    break
        except Exception:
            pass
        change_batch = json.dumps(
            {
                "Changes": [
                    {
                        "Action": "UPSERT",
                        "ResourceRecordSet": {
                            "Name": "api.example.com",
                            "Type": "A",
                            "TTL": 300,
                            "ResourceRecords": [{"Value": "10.0.1.50"}],
                        },
                    }
                ]
            }
        )
        cmds.append(
            f"aws route53 change-resource-record-sets --hosted-zone-id {zone_id} --change-batch '{change_batch}'"
        )

    elif task_id == 115:
        ok, out, _ = backend.execute_command(
            "aws elbv2 describe-target-groups --names web-targets"
        )
        tg_arn = "UNKNOWN"
        try:
            tg_arn = json.loads(out)["TargetGroups"][0]["TargetGroupArn"]
        except Exception:
            pass
        cmds.append(
            f"aws elbv2 modify-target-group --target-group-arn {tg_arn} --health-check-path /health --health-check-port 80 --health-check-interval-seconds 15 --healthy-threshold-count 2"
        )

    elif task_id == 126:
        ok, out, _ = backend.execute_command(
            "aws cognito-idp list-user-pools --max-results 10"
        )
        pool_id = "UNKNOWN"
        try:
            for pool in json.loads(out).get("UserPools", []):
                if "customer-auth" in pool.get("Name", ""):
                    pool_id = pool["Id"]
                    break
        except Exception:
            pass
        policies = json.dumps(
            {
                "PasswordPolicy": {
                    "MinimumLength": 12,
                    "RequireUppercase": True,
                    "RequireLowercase": True,
                    "RequireNumbers": True,
                    "RequireSymbols": True,
                    "TemporaryPasswordValidityDays": 1,
                }
            }
        )
        cmds.append(
            f"aws cognito-idp update-user-pool --user-pool-id {pool_id} --policies '{policies}'"
        )

    return cmds


# ---------------------------------------------------------------------------
# Intermediate tasks with dynamic follow-ups
# ---------------------------------------------------------------------------


def _intermediate_dynamic(
    task_id: int, backend: EnvironmentStrategy, step: int, static_cmds: list[str]
) -> list[str]:
    """Resolve dynamic follow-up commands for intermediate tasks."""
    cmds = list(static_cmds)

    if task_id == 76 and step >= 1:
        ok, out, _ = backend.execute_command(
            "aws cognito-idp list-user-pools --max-results 10"
        )
        pool_id = "UNKNOWN"
        try:
            for pool in json.loads(out).get("UserPools", []):
                if "app-users" in pool.get("Name", ""):
                    pool_id = pool["Id"]
                    break
        except Exception:
            pass
        cmds.append(
            f"aws cognito-idp create-user-pool-client --user-pool-id {pool_id} --client-name web-app-client"
        )

    elif task_id == 78 and step >= 1:
        ok, out, _ = backend.execute_command("aws ec2 describe-volumes")
        vol_id = "UNKNOWN"
        try:
            for vol in json.loads(out).get("Volumes", []):
                vol_id = vol["VolumeId"]
                break
        except Exception:
            pass
        cmds.append(
            f"aws ec2 create-tags --resources {vol_id} --tags Key=Name,Value=data-volume"
        )

    elif task_id == 82 and step >= 1:
        ok, out, _ = backend.execute_command("aws apigatewayv2 get-apis")
        api_id = "UNKNOWN"
        try:
            for api in json.loads(out).get("Items", []):
                if "products-api" in api.get("Name", ""):
                    api_id = api["ApiId"]
                    break
        except Exception:
            pass
        cmds.append(
            f'aws apigatewayv2 create-route --api-id {api_id} --route-key "GET /products-api"'
        )

    elif task_id == 84 and step >= 1:
        ok, out, _ = backend.execute_command(
            "aws sqs get-queue-url --queue-name task-queue"
        )
        queue_url = "UNKNOWN"
        try:
            queue_url = json.loads(out)["QueueUrl"]
        except Exception:
            pass
        cmds.append(
            f"""aws sqs send-message --queue-url {queue_url} --message-body '{{"task":"process","id":"task-queue-001"}}'"""
        )

    return cmds


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

_ADVANCED_IDS = {
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
}
_INTERMEDIATE_DYNAMIC_IDS = {76, 78, 82, 84}
_EXPERT_DYNAMIC_IDS = {114, 115, 126}


def get_next_solution(
    task_id: int,
    backend: EnvironmentStrategy,
    tracker: EpisodeTracker,
) -> dict:
    """Return the next solution command for the given task.

    Returns:
        {"command": str | None, "step": int, "total_steps": int}
    """
    step = tracker.step_count

    # Advanced: fully dynamic command sequences
    if task_id in _ADVANCED_IDS:
        cmds = _advanced_commands(task_id, backend, step)
        if step < len(cmds):
            return {"command": cmds[step], "step": step + 1, "total_steps": len(cmds)}
        return {"command": None, "step": step, "total_steps": len(cmds)}

    # Load static commands
    static = _load_static()
    base_cmds = static.get(task_id, [])

    # Intermediate with dynamic follow-ups
    if task_id in _INTERMEDIATE_DYNAMIC_IDS:
        cmds = _intermediate_dynamic(task_id, backend, step, base_cmds)
        if step < len(cmds):
            return {"command": cmds[step], "step": step + 1, "total_steps": len(cmds)}
        return {"command": None, "step": step, "total_steps": len(cmds)}

    # Expert with dynamic IDs
    if task_id in _EXPERT_DYNAMIC_IDS:
        cmds = _expert_dynamic_command(task_id, backend, step, base_cmds)
        if step < len(cmds):
            return {"command": cmds[step], "step": step + 1, "total_steps": len(cmds)}
        return {"command": None, "step": step, "total_steps": len(cmds)}

    # Default: static commands
    if step < len(base_cmds):
        return {
            "command": base_cmds[step],
            "step": step + 1,
            "total_steps": len(base_cmds),
        }
    return {"command": None, "step": step, "total_steps": len(base_cmds)}
