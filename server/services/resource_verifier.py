"""Resource verification service — queries MiniStack for ground-truth state."""

from __future__ import annotations

import json
import logging
from typing import Any

from server.services.environment_strategy import EnvironmentStrategy

logger = logging.getLogger(__name__)


def _extract_json_path(data: Any, path: str) -> Any:
    """Simple JSON path extractor supporting dot notation and array indexing.

    Supports paths like: $.Table.ProvisionedThroughput.ReadCapacityUnits
                         $.Rules[0].Expiration.Days
                         $.Buckets[].Name
    """
    parts = path.lstrip("$").lstrip(".").split(".")
    current = data
    for part in parts:
        if current is None:
            return None
        # Handle array index like Rules[0]
        if "[" in part:
            key, idx_str = part.split("[", 1)
            idx_str = idx_str.rstrip("]")
            if key:
                current = current.get(key) if isinstance(current, dict) else None
            if current is None:
                return None
            if idx_str == "":
                # Wildcard — return list of values
                if isinstance(current, list):
                    remaining = ".".join(parts[parts.index(part) + 1 :])
                    if remaining:
                        return [
                            _extract_json_path(item, f"$.{remaining}")
                            for item in current
                        ]
                    return current
                return None
            try:
                current = current[int(idx_str)]
            except (IndexError, TypeError):
                return None
        else:
            current = current.get(part) if isinstance(current, dict) else None
    return current


class ResourceVerifier:
    """Verifies resource state by querying MiniStack via AWS CLI."""

    def __init__(self, backend: EnvironmentStrategy) -> None:
        self._backend = backend

    def resource_exists(self, service: str, name: str) -> bool:
        """Check if a specific resource exists in MiniStack.

        Uses service-specific verification commands and checks for the
        exact resource name (not just any resource of that type).
        """
        service_lower = service.lower()
        verifiers = {
            "s3": self._check_s3_bucket,
            "dynamodb": self._check_dynamodb_table,
            "lambda": self._check_lambda_function,
            "sqs": self._check_sqs_queue,
            "sns": self._check_sns_topic,
            "iam": self._check_iam_resource,
            "apigateway": self._check_apigateway,
            "secretsmanager": self._check_secretsmanager,
            "ecs": self._check_ecs_cluster,
            "rds": self._check_rds_instance,
            "elasticache": self._check_elasticache_cluster,
            "route53": self._check_route53_hosted_zone,
            "elbv2": self._check_elbv2_load_balancer,
            "efs": self._check_efs_filesystem,
            "cognito-idp": self._check_cognito_user_pool,
            "ssm": self._check_ssm_parameter,
            "events": self._check_eventbridge_rule,
            "apigatewayv2": self._check_apigatewayv2,
            "cloudformation": self._check_cloudformation_stack,
            "glue": self._check_glue_database,
            "ebs": self._check_ebs_volume,
            "firehose": self._check_firehose_stream,
        }
        verifier = verifiers.get(service_lower)
        if verifier is None:
            logger.warning("No verifier for service: %s", service)
            return False
        return verifier(name)

    def check_state(self, state_check: dict[str, Any]) -> bool:
        """Run an arbitrary command and assert on its output.

        Supports:
          - output_contains: substring check on stdout
          - json_path + expected: extract value from JSON stdout and compare
        """
        command = state_check.get("command", "")
        if not command:
            return False

        success, stdout, _ = self._backend.execute_command(command)
        if not success:
            return False

        # Check output_contains
        if "output_contains" in state_check:
            if state_check["output_contains"] not in stdout:
                return False

        # Check json_path + expected
        if "json_path" in state_check and "expected" in state_check:
            try:
                data = json.loads(stdout)
                value = _extract_json_path(data, state_check["json_path"])
                expected = state_check["expected"]
                # Compare as strings for flexibility
                if str(value) != str(expected):
                    return False
            except (json.JSONDecodeError, KeyError, TypeError):
                return False

        return True

    # -- Service-specific verifiers -------------------------------------------

    def _check_s3_bucket(self, name: str) -> bool:
        success, stdout, _ = self._backend.execute_command(
            "aws s3api list-buckets --output json"
        )
        if not success:
            return False
        try:
            data = json.loads(stdout)
            buckets = data.get("Buckets", [])
            return any(b.get("Name") == name for b in buckets)
        except (json.JSONDecodeError, TypeError):
            return False

    def _check_dynamodb_table(self, name: str) -> bool:
        success, _, _ = self._backend.execute_command(
            f"aws dynamodb describe-table --table-name {name}"
        )
        return success

    def _check_lambda_function(self, name: str) -> bool:
        success, _, _ = self._backend.execute_command(
            f"aws lambda get-function --function-name {name}"
        )
        return success

    def _check_sqs_queue(self, name: str) -> bool:
        success, _, _ = self._backend.execute_command(
            f"aws sqs get-queue-url --queue-name {name}"
        )
        return success

    def _check_sns_topic(self, name: str) -> bool:
        success, stdout, _ = self._backend.execute_command(
            "aws sns list-topics --output json"
        )
        if not success:
            return False
        try:
            data = json.loads(stdout)
            topics = data.get("Topics", [])
            return any(name in t.get("TopicArn", "") for t in topics)
        except (json.JSONDecodeError, TypeError):
            return False

    def _check_iam_resource(self, name: str) -> bool:
        """Check for IAM roles, users, and policies by name."""
        # Try role first
        success, _, _ = self._backend.execute_command(
            f"aws iam get-role --role-name {name}"
        )
        if success:
            return True
        # Try user
        success, _, _ = self._backend.execute_command(
            f"aws iam get-user --user-name {name}"
        )
        if success:
            return True
        # Try policy (list and match by name)
        success, stdout, _ = self._backend.execute_command(
            "aws iam list-policies --scope Local --output json"
        )
        if success:
            try:
                data = json.loads(stdout)
                policies = data.get("Policies", [])
                if any(p.get("PolicyName") == name for p in policies):
                    return True
            except (json.JSONDecodeError, TypeError):
                pass
        return False

    def _check_secretsmanager(self, name: str) -> bool:
        success, _, _ = self._backend.execute_command(
            f"aws secretsmanager describe-secret --secret-id {name}"
        )
        return success

    def _check_apigateway(self, name: str) -> bool:
        success, stdout, _ = self._backend.execute_command(
            "aws apigateway get-rest-apis --output json"
        )
        if not success:
            return False
        try:
            data = json.loads(stdout)
            items = data.get("items", [])
            return any(i.get("name") == name for i in items)
        except (json.JSONDecodeError, TypeError):
            return False

    def _check_ecs_cluster(self, name: str) -> bool:
        success, stdout, _ = self._backend.execute_command(
            f"aws ecs describe-clusters --clusters {name}"
        )
        if not success:
            return False
        try:
            data = json.loads(stdout)
            clusters = data.get("clusters", [])
            return any(
                c.get("clusterName") == name and c.get("status") != "INACTIVE"
                for c in clusters
            )
        except (json.JSONDecodeError, TypeError):
            return False

    def _check_rds_instance(self, name: str) -> bool:
        success, _, _ = self._backend.execute_command(
            f"aws rds describe-db-instances --db-instance-identifier {name}"
        )
        return success

    def _check_elasticache_cluster(self, name: str) -> bool:
        success, _, _ = self._backend.execute_command(
            f"aws elasticache describe-cache-clusters --cache-cluster-id {name}"
        )
        return success

    def _check_route53_hosted_zone(self, name: str) -> bool:
        success, stdout, _ = self._backend.execute_command(
            "aws route53 list-hosted-zones --output json"
        )
        if not success:
            return False
        try:
            data = json.loads(stdout)
            zones = data.get("HostedZones", [])
            return any(z.get("Name", "").rstrip(".") == name.rstrip(".") for z in zones)
        except (json.JSONDecodeError, TypeError):
            return False

    def _check_elbv2_load_balancer(self, name: str) -> bool:
        success, stdout, _ = self._backend.execute_command(
            f"aws elbv2 describe-load-balancers --names {name}"
        )
        if not success:
            return False
        try:
            data = json.loads(stdout)
            lbs = data.get("LoadBalancers", [])
            return any(lb.get("LoadBalancerName") == name for lb in lbs)
        except (json.JSONDecodeError, TypeError):
            return False

    def _check_efs_filesystem(self, name: str) -> bool:
        success, stdout, _ = self._backend.execute_command(
            "aws efs describe-file-systems --output json"
        )
        if not success:
            return False
        try:
            data = json.loads(stdout)
            filesystems = data.get("FileSystems", [])
            return any(
                fs.get("CreationToken") == name
                or any(t.get("Value") == name for t in fs.get("Tags", []))
                for fs in filesystems
            )
        except (json.JSONDecodeError, TypeError):
            return False

    def _check_cognito_user_pool(self, name: str) -> bool:
        success, stdout, _ = self._backend.execute_command(
            "aws cognito-idp list-user-pools --max-results 60 --output json"
        )
        if not success:
            return False
        try:
            data = json.loads(stdout)
            pools = data.get("UserPools", [])
            return any(p.get("Name") == name for p in pools)
        except (json.JSONDecodeError, TypeError):
            return False

    def _check_ssm_parameter(self, name: str) -> bool:
        success, _, _ = self._backend.execute_command(
            f"aws ssm get-parameter --name {name}"
        )
        return success

    def _check_eventbridge_rule(self, name: str) -> bool:
        success, _, _ = self._backend.execute_command(
            f"aws events describe-rule --name {name}"
        )
        return success

    def _check_apigatewayv2(self, name: str) -> bool:
        success, stdout, _ = self._backend.execute_command(
            "aws apigatewayv2 get-apis --output json"
        )
        if not success:
            return False
        try:
            data = json.loads(stdout)
            items = data.get("Items", [])
            return any(i.get("Name") == name for i in items)
        except (json.JSONDecodeError, TypeError):
            return False

    def _check_cloudformation_stack(self, name: str) -> bool:
        success, _, _ = self._backend.execute_command(
            f"aws cloudformation describe-stacks --stack-name {name}"
        )
        return success

    def _check_glue_database(self, name: str) -> bool:
        success, _, _ = self._backend.execute_command(
            f"aws glue get-database --name {name}"
        )
        return success

    def _check_ebs_volume(self, name: str) -> bool:
        success, stdout, _ = self._backend.execute_command(
            "aws ec2 describe-volumes --output json"
        )
        if not success:
            return False
        try:
            data = json.loads(stdout)
            volumes = data.get("Volumes", [])
            return len(volumes) > 0
        except (json.JSONDecodeError, TypeError):
            return False

    def _check_firehose_stream(self, name: str) -> bool:
        success, _, _ = self._backend.execute_command(
            f"aws firehose describe-delivery-stream --delivery-stream-name {name}"
        )
        return success
