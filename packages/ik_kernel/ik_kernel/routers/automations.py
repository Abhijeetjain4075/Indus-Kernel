import uuid

from fastapi import APIRouter
from ik_automation import Automation
from pydantic import BaseModel, Field

router = APIRouter()
_items = {}


class AutomationRequest(BaseModel):
    trigger: str = Field(min_length=1)
    action: str = Field(min_length=1)


@router.get("")
async def list_automations():
    return {"automations": [a.__dict__ for a in _items.values()]}


@router.post("")
async def create_automation(req: AutomationRequest):
    a = Automation(str(uuid.uuid4()), req.trigger, req.action)
    _items[a.id] = a
    return a.__dict__
