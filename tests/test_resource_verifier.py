"""Unit tests for ResourceVerifier — resource existence checks, state checks, and JSON path extraction.

Uses a mock AwsBackend so tests run without MiniStack/Docker.

Run:
    python -m pytest tests/test_resource_verifier.py -v
"""

from __future__ import annotations

import json
from unittest.mock import MagicMock


from server.services.environment_strategy import EnvironmentStrategy
from server.services.resource_verifier import ResourceVerifier, _extract_json_path


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _mock_backend(responses: dict[str, tuple[bool, str, str]]) -> EnvironmentStrategy:
    """Create a mock AwsBackend that returns preset responses keyed by substring match."""
    backend = MagicMock(spec=EnvironmentStrategy)

    def execute(cmd: str) -> tuple[bool, str, str]:
        for pattern, result in responses.items():
            if pattern in cmd:
                return result
        return (False, "", "unknown command")

    backend.execute_command.side_effect = execute
    return backend


# ---------------------------------------------------------------------------
# _extract_json_path
# ---------------------------------------------------------------------------


class TestExtractJsonPath:
    def test_simple_dot_path(self) -> None:
        data = {"Table": {"Name": "orders"}}
        assert _extract_json_path(data, "$.Table.Name") == "orders"

    def test_nested_numeric(self) -> None:
        data = {"Table": {"ProvisionedThroughput": {"ReadCapacityUnits": 50}}}
        assert (
            _extract_json_path(data, "$.Table.ProvisionedThroughput.ReadCapacityUnits")
            == 50
        )

    def test_array_index(self) -> None:
        data = {"Rules": [{"ID": "first"}, {"ID": "second"}]}
        assert _extract_json_path(data, "$.Rules[0].ID") == "first"
        assert _extract_json_path(data, "$.Rules[1].ID") == "second"

    def test_array_index_out_of_bounds(self) -> None:
        data = {"Rules": [{"ID": "only"}]}
        assert _extract_json_path(data, "$.Rules[5].ID") is None

    def test_wildcard_array(self) -> None:
        data = {"Buckets": [{"Name": "a"}, {"Name": "b"}]}
        assert _extract_json_path(data, "$.Buckets[].Name") == ["a", "b"]

    def test_wildcard_no_remaining(self) -> None:
        data = {"Items": [1, 2, 3]}
        assert _extract_json_path(data, "$.Items[]") == [1, 2, 3]

    def test_missing_key(self) -> None:
        assert _extract_json_path({"a": 1}, "$.b.c") is None

    def test_none_data(self) -> None:
        assert _extract_json_path(None, "$.foo") is None

    def test_non_dict_intermediate(self) -> None:
        data = {"a": "string_not_dict"}
        assert _extract_json_path(data, "$.a.b") is None

    def test_services_nested_path(self) -> None:
        data = {"services": [{"desiredCount": 3}]}
        assert _extract_json_path(data, "$.services[0].desiredCount") == 3

    def test_attributes_path(self) -> None:
        data = {"Attributes": {"VisibilityTimeout": "120"}}
        assert _extract_json_path(data, "$.Attributes.VisibilityTimeout") == "120"


# ---------------------------------------------------------------------------
# ResourceVerifier.check_state
# ---------------------------------------------------------------------------


class TestCheckState:
    def test_output_contains_pass(self) -> None:
        backend = _mock_backend({"list-attached": (True, "AmazonSQSFullAccess", "")})
        v = ResourceVerifier(backend)
        assert v.check_state(
            {"command": "aws iam list-attached-role-policies", "output_contains": "SQS"}
        )

    def test_output_contains_fail(self) -> None:
        backend = _mock_backend({"list-attached": (True, "AmazonS3ReadOnly", "")})
        v = ResourceVerifier(backend)
        assert not v.check_state(
            {"command": "aws iam list-attached-role-policies", "output_contains": "SQS"}
        )

    def test_command_fails(self) -> None:
        backend = _mock_backend({"describe": (False, "", "not found")})
        v = ResourceVerifier(backend)
        assert not v.check_state(
            {"command": "aws describe-something", "output_contains": "ok"}
        )

    def test_empty_command(self) -> None:
        backend = _mock_backend({})
        v = ResourceVerifier(backend)
        assert not v.check_state({"command": ""})
        assert not v.check_state({})

    def test_json_path_expected(self) -> None:
        stdout = json.dumps(
            {"Table": {"ProvisionedThroughput": {"ReadCapacityUnits": 50}}}
        )
        backend = _mock_backend({"describe-table": (True, stdout, "")})
        v = ResourceVerifier(backend)
        assert v.check_state(
            {
                "command": "aws dynamodb describe-table --table-name t",
                "json_path": "$.Table.ProvisionedThroughput.ReadCapacityUnits",
                "expected": 50,
            }
        )

    def test_json_path_string_comparison(self) -> None:
        stdout = json.dumps({"Attributes": {"VisibilityTimeout": "120"}})
        backend = _mock_backend({"get-queue": (True, stdout, "")})
        v = ResourceVerifier(backend)
        assert v.check_state(
            {
                "command": "aws sqs get-queue-attributes",
                "json_path": "$.Attributes.VisibilityTimeout",
                "expected": "120",
            }
        )

    def test_json_path_mismatch(self) -> None:
        stdout = json.dumps(
            {"Table": {"ProvisionedThroughput": {"ReadCapacityUnits": 5}}}
        )
        backend = _mock_backend({"describe-table": (True, stdout, "")})
        v = ResourceVerifier(backend)
        assert not v.check_state(
            {
                "command": "aws dynamodb describe-table",
                "json_path": "$.Table.ProvisionedThroughput.ReadCapacityUnits",
                "expected": 50,
            }
        )

    def test_json_path_invalid_json(self) -> None:
        backend = _mock_backend({"describe": (True, "not-json{", "")})
        v = ResourceVerifier(backend)
        assert not v.check_state(
            {
                "command": "aws describe-something",
                "json_path": "$.foo",
                "expected": "bar",
            }
        )

    def test_both_output_contains_and_json_path(self) -> None:
        stdout = json.dumps({"Timeout": 30, "FunctionName": "payment-webhook"})
        backend = _mock_backend({"get-function": (True, stdout, "")})
        v = ResourceVerifier(backend)
        # Both checks must pass
        assert v.check_state(
            {
                "command": "aws lambda get-function-configuration",
                "output_contains": "payment-webhook",
                "json_path": "$.Timeout",
                "expected": 30,
            }
        )

    def test_output_contains_pass_json_path_fail(self) -> None:
        stdout = json.dumps({"Timeout": 3, "FunctionName": "payment-webhook"})
        backend = _mock_backend({"get-function": (True, stdout, "")})
        v = ResourceVerifier(backend)
        assert not v.check_state(
            {
                "command": "aws lambda get-function-configuration",
                "output_contains": "payment-webhook",
                "json_path": "$.Timeout",
                "expected": 30,
            }
        )

    def test_only_json_path_no_expected_still_passes(self) -> None:
        # json_path without expected is not evaluated
        backend = _mock_backend({"cmd": (True, '{"a":1}', "")})
        v = ResourceVerifier(backend)
        assert v.check_state({"command": "aws cmd", "json_path": "$.a"})


# ---------------------------------------------------------------------------
# ResourceVerifier.resource_exists — service verifiers
# ---------------------------------------------------------------------------


class TestResourceExistsS3:
    def test_bucket_exists(self) -> None:
        stdout = json.dumps({"Buckets": [{"Name": "my-bucket"}, {"Name": "other"}]})
        backend = _mock_backend({"list-buckets": (True, stdout, "")})
        v = ResourceVerifier(backend)
        assert v.resource_exists("s3", "my-bucket")

    def test_bucket_missing(self) -> None:
        stdout = json.dumps({"Buckets": [{"Name": "other"}]})
        backend = _mock_backend({"list-buckets": (True, stdout, "")})
        v = ResourceVerifier(backend)
        assert not v.resource_exists("s3", "my-bucket")

    def test_list_fails(self) -> None:
        backend = _mock_backend({"list-buckets": (False, "", "err")})
        v = ResourceVerifier(backend)
        assert not v.resource_exists("s3", "demo")


class TestResourceExistsDynamoDB:
    def test_table_exists(self) -> None:
        backend = _mock_backend({"describe-table": (True, "{}", "")})
        v = ResourceVerifier(backend)
        assert v.resource_exists("dynamodb", "orders")

    def test_table_missing(self) -> None:
        backend = _mock_backend({"describe-table": (False, "", "not found")})
        v = ResourceVerifier(backend)
        assert not v.resource_exists("dynamodb", "orders")


class TestResourceExistsLambda:
    def test_function_exists(self) -> None:
        backend = _mock_backend({"get-function": (True, "{}", "")})
        v = ResourceVerifier(backend)
        assert v.resource_exists("lambda", "processor")

    def test_function_missing(self) -> None:
        backend = _mock_backend({"get-function": (False, "", "not found")})
        v = ResourceVerifier(backend)
        assert not v.resource_exists("lambda", "processor")


class TestResourceExistsSQS:
    def test_queue_exists(self) -> None:
        backend = _mock_backend({"get-queue-url": (True, "http://...", "")})
        v = ResourceVerifier(backend)
        assert v.resource_exists("sqs", "my-queue")

    def test_queue_missing(self) -> None:
        backend = _mock_backend({"get-queue-url": (False, "", "not found")})
        v = ResourceVerifier(backend)
        assert not v.resource_exists("sqs", "my-queue")


class TestResourceExistsSNS:
    def test_topic_exists(self) -> None:
        stdout = json.dumps(
            {"Topics": [{"TopicArn": "arn:aws:sns:us-east-1:000000000000:alerts"}]}
        )
        backend = _mock_backend({"list-topics": (True, stdout, "")})
        v = ResourceVerifier(backend)
        assert v.resource_exists("sns", "alerts")

    def test_topic_missing(self) -> None:
        stdout = json.dumps(
            {"Topics": [{"TopicArn": "arn:aws:sns:us-east-1:000000000000:other"}]}
        )
        backend = _mock_backend({"list-topics": (True, stdout, "")})
        v = ResourceVerifier(backend)
        assert not v.resource_exists("sns", "alerts")


class TestResourceExistsIAM:
    def test_role_exists(self) -> None:
        backend = _mock_backend({"get-role": (True, "{}", "")})
        v = ResourceVerifier(backend)
        assert v.resource_exists("iam", "my-role")

    def test_user_exists(self) -> None:
        backend = _mock_backend(
            {"get-role": (False, "", ""), "get-user": (True, "{}", "")}
        )
        v = ResourceVerifier(backend)
        assert v.resource_exists("iam", "deploy-bot")

    def test_policy_exists(self) -> None:
        stdout = json.dumps({"Policies": [{"PolicyName": "my-policy"}]})
        backend = _mock_backend(
            {
                "get-role": (False, "", ""),
                "get-user": (False, "", ""),
                "list-policies": (True, stdout, ""),
            }
        )
        v = ResourceVerifier(backend)
        assert v.resource_exists("iam", "my-policy")

    def test_iam_not_found(self) -> None:
        backend = _mock_backend(
            {
                "get-role": (False, "", ""),
                "get-user": (False, "", ""),
                "list-policies": (True, json.dumps({"Policies": []}), ""),
            }
        )
        v = ResourceVerifier(backend)
        assert not v.resource_exists("iam", "ghost")


class TestResourceExistsSecretsManager:
    def test_secret_exists(self) -> None:
        backend = _mock_backend({"describe-secret": (True, "{}", "")})
        v = ResourceVerifier(backend)
        assert v.resource_exists("secretsmanager", "db-creds")

    def test_secret_missing(self) -> None:
        backend = _mock_backend({"describe-secret": (False, "", "not found")})
        v = ResourceVerifier(backend)
        assert not v.resource_exists("secretsmanager", "db-creds")


class TestResourceExistsApiGateway:
    def test_api_exists(self) -> None:
        stdout = json.dumps({"items": [{"name": "my-api"}]})
        backend = _mock_backend({"get-rest-apis": (True, stdout, "")})
        v = ResourceVerifier(backend)
        assert v.resource_exists("apigateway", "my-api")

    def test_api_missing(self) -> None:
        stdout = json.dumps({"items": []})
        backend = _mock_backend({"get-rest-apis": (True, stdout, "")})
        v = ResourceVerifier(backend)
        assert not v.resource_exists("apigateway", "my-api")


class TestResourceExistsECS:
    def test_cluster_exists_active(self) -> None:
        stdout = json.dumps({"clusters": [{"clusterName": "prod", "status": "ACTIVE"}]})
        backend = _mock_backend({"describe-clusters": (True, stdout, "")})
        v = ResourceVerifier(backend)
        assert v.resource_exists("ecs", "prod")

    def test_cluster_inactive(self) -> None:
        stdout = json.dumps(
            {"clusters": [{"clusterName": "prod", "status": "INACTIVE"}]}
        )
        backend = _mock_backend({"describe-clusters": (True, stdout, "")})
        v = ResourceVerifier(backend)
        assert not v.resource_exists("ecs", "prod")

    def test_cluster_not_found(self) -> None:
        backend = _mock_backend({"describe-clusters": (False, "", "")})
        v = ResourceVerifier(backend)
        assert not v.resource_exists("ecs", "prod")


class TestResourceExistsRDS:
    def test_instance_exists(self) -> None:
        backend = _mock_backend({"describe-db-instances": (True, "{}", "")})
        v = ResourceVerifier(backend)
        assert v.resource_exists("rds", "my-db")

    def test_instance_missing(self) -> None:
        backend = _mock_backend({"describe-db-instances": (False, "", "not found")})
        v = ResourceVerifier(backend)
        assert not v.resource_exists("rds", "my-db")


class TestResourceExistsElastiCache:
    def test_cluster_exists(self) -> None:
        backend = _mock_backend({"describe-cache-clusters": (True, "{}", "")})
        v = ResourceVerifier(backend)
        assert v.resource_exists("elasticache", "session-cache")


class TestResourceExistsRoute53:
    def test_zone_exists(self) -> None:
        stdout = json.dumps({"HostedZones": [{"Name": "example.com."}]})
        backend = _mock_backend({"list-hosted-zones": (True, stdout, "")})
        v = ResourceVerifier(backend)
        assert v.resource_exists("route53", "example.com")

    def test_zone_trailing_dot_normalized(self) -> None:
        stdout = json.dumps({"HostedZones": [{"Name": "example.com."}]})
        backend = _mock_backend({"list-hosted-zones": (True, stdout, "")})
        v = ResourceVerifier(backend)
        assert v.resource_exists("route53", "example.com.")


class TestResourceExistsELBv2:
    def test_lb_exists(self) -> None:
        stdout = json.dumps({"LoadBalancers": [{"LoadBalancerName": "web-alb"}]})
        backend = _mock_backend({"describe-load-balancers": (True, stdout, "")})
        v = ResourceVerifier(backend)
        assert v.resource_exists("elbv2", "web-alb")

    def test_lb_missing(self) -> None:
        backend = _mock_backend({"describe-load-balancers": (False, "", "not found")})
        v = ResourceVerifier(backend)
        assert not v.resource_exists("elbv2", "web-alb")


class TestResourceExistsEFS:
    def test_fs_by_creation_token(self) -> None:
        stdout = json.dumps(
            {"FileSystems": [{"CreationToken": "app-storage", "Tags": []}]}
        )
        backend = _mock_backend({"describe-file-systems": (True, stdout, "")})
        v = ResourceVerifier(backend)
        assert v.resource_exists("efs", "app-storage")

    def test_fs_by_tag(self) -> None:
        stdout = json.dumps(
            {
                "FileSystems": [
                    {
                        "CreationToken": "token-123",
                        "Tags": [{"Key": "Name", "Value": "shared-data"}],
                    }
                ]
            }
        )
        backend = _mock_backend({"describe-file-systems": (True, stdout, "")})
        v = ResourceVerifier(backend)
        assert v.resource_exists("efs", "shared-data")

    def test_fs_missing(self) -> None:
        stdout = json.dumps({"FileSystems": []})
        backend = _mock_backend({"describe-file-systems": (True, stdout, "")})
        v = ResourceVerifier(backend)
        assert not v.resource_exists("efs", "nonexistent")


class TestResourceExistsCognito:
    def test_pool_exists(self) -> None:
        stdout = json.dumps({"UserPools": [{"Name": "customer-auth"}]})
        backend = _mock_backend({"list-user-pools": (True, stdout, "")})
        v = ResourceVerifier(backend)
        assert v.resource_exists("cognito-idp", "customer-auth")

    def test_pool_missing(self) -> None:
        stdout = json.dumps({"UserPools": []})
        backend = _mock_backend({"list-user-pools": (True, stdout, "")})
        v = ResourceVerifier(backend)
        assert not v.resource_exists("cognito-idp", "customer-auth")


class TestResourceExistsSSM:
    def test_param_exists(self) -> None:
        backend = _mock_backend({"get-parameter": (True, "{}", "")})
        v = ResourceVerifier(backend)
        assert v.resource_exists("ssm", "/app/config")


class TestResourceExistsEventBridge:
    def test_rule_exists(self) -> None:
        backend = _mock_backend({"describe-rule": (True, "{}", "")})
        v = ResourceVerifier(backend)
        assert v.resource_exists("events", "nightly-etl")


class TestResourceExistsApiGatewayV2:
    def test_api_exists(self) -> None:
        stdout = json.dumps({"Items": [{"Name": "products-api"}]})
        backend = _mock_backend({"get-apis": (True, stdout, "")})
        v = ResourceVerifier(backend)
        assert v.resource_exists("apigatewayv2", "products-api")

    def test_api_missing(self) -> None:
        stdout = json.dumps({"Items": []})
        backend = _mock_backend({"get-apis": (True, stdout, "")})
        v = ResourceVerifier(backend)
        assert not v.resource_exists("apigatewayv2", "products-api")


class TestResourceExistsCloudFormation:
    def test_stack_exists(self) -> None:
        backend = _mock_backend({"describe-stacks": (True, "{}", "")})
        v = ResourceVerifier(backend)
        assert v.resource_exists("cloudformation", "vpc-stack")


class TestResourceExistsGlue:
    def test_database_exists(self) -> None:
        backend = _mock_backend({"get-database": (True, "{}", "")})
        v = ResourceVerifier(backend)
        assert v.resource_exists("glue", "analytics-db")


class TestResourceExistsEBS:
    def test_volume_exists(self) -> None:
        stdout = json.dumps({"Volumes": [{"VolumeId": "vol-123"}]})
        backend = _mock_backend({"describe-volumes": (True, stdout, "")})
        v = ResourceVerifier(backend)
        assert v.resource_exists("ebs", "data-volume")

    def test_no_volumes(self) -> None:
        stdout = json.dumps({"Volumes": []})
        backend = _mock_backend({"describe-volumes": (True, stdout, "")})
        v = ResourceVerifier(backend)
        assert not v.resource_exists("ebs", "data-volume")


class TestResourceExistsFirehose:
    def test_stream_exists(self) -> None:
        backend = _mock_backend({"describe-delivery-stream": (True, "{}", "")})
        v = ResourceVerifier(backend)
        assert v.resource_exists("firehose", "event-stream")


class TestResourceExistsUnknownService:
    def test_unknown_service(self) -> None:
        backend = _mock_backend({})
        v = ResourceVerifier(backend)
        assert not v.resource_exists("unknown-service", "name")


class TestResourceExistsInvalidJson:
    def test_s3_bad_json(self) -> None:
        backend = _mock_backend({"list-buckets": (True, "not-json", "")})
        v = ResourceVerifier(backend)
        assert not v.resource_exists("s3", "demo")

    def test_sns_bad_json(self) -> None:
        backend = _mock_backend({"list-topics": (True, "{bad", "")})
        v = ResourceVerifier(backend)
        assert not v.resource_exists("sns", "alerts")
