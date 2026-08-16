"""Optional Temporal adapter with explicit workflow registration boundary."""


class TemporalUnavailable(RuntimeError):
    pass


class TemporalWorkflowClient:
    def __init__(self, target="localhost:7233", namespace="default"):
        self.target = target
        self.namespace = namespace
        self.client = None

    async def connect(self):
        try:
            from temporalio.client import Client
        except ImportError as exc:
            raise TemporalUnavailable("install temporalio") from exc
        self.client = await Client.connect(self.target, namespace=self.namespace)
        return self

    async def start(self, workflow, *args, task_queue="indus"):
        if self.client is None:
            raise TemporalUnavailable("connect() required")
        return await self.client.start_workflow(
            workflow, *args, id=getattr(workflow, "__name__", "workflow"), task_queue=task_queue
        )
