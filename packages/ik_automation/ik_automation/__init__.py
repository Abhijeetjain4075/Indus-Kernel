from dataclasses import dataclass
from typing import Callable
@dataclass(frozen=True)
class Automation:
 id:str; trigger:str; action:str; enabled:bool=True
class AutomationEngine:
 def __init__(self): self._items={}; self._handlers={}
 def register(self,a:Automation,handler:Callable):
  if a.id in self._items: raise ValueError("automation already registered")
  self._items[a.id]=a; self._handlers[a.id]=handler
 def trigger(self,event:str):
  results=[]
  for aid,a in self._items.items():
   if a.enabled and a.trigger==event: results.append(self._handlers[aid](event))
  return results
