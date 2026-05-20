from pydantic import BaseModel


class Topology(BaseModel):
    replicas: int = 1
    nodes_per_replica: int = 1
