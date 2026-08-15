from dataclasses import dataclass
@dataclass(frozen=True)
class CodeTask:
    language:str; instruction:str
def validate_code_request(task:CodeTask)->None:
    if not task.language or not task.instruction.strip(): raise ValueError("language and instruction are required")
