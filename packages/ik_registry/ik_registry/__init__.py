from dataclasses import dataclass
@dataclass(frozen=True)
class ModelRecord:
    id:str; version:str; provider:str; status:str="active"
class ModelRegistry:
    def __init__(self): self._models={}
    def register(self,record:ModelRecord): self._models[record.id]=record
    def get(self,model_id): return self._models.get(model_id)
    def list(self): return list(self._models.values())
registry=ModelRegistry()
