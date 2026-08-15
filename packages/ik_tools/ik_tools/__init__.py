from dataclasses import dataclass
from typing import Callable, Any
@dataclass(frozen=True)
class ToolSpec:
    name:str; description:str; risk:str="low"
class ToolRegistry:
    def __init__(self): self._tools:dict[str,tuple[ToolSpec,Callable[...,Any]]]={}
    def register(self,spec:ToolSpec,fn:Callable[...,Any])->None:
        if spec.name in self._tools: raise ValueError(f"tool already registered: {spec.name}")
        self._tools[spec.name]=(spec,fn)
    def list(self): return [s for s,_ in self._tools.values()]
    def call(self,name:str,**kwargs): 
        if name not in self._tools: raise KeyError(name)
        return self._tools[name][1](**kwargs)
registry=ToolRegistry()
