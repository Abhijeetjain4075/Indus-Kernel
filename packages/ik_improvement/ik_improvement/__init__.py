from dataclasses import dataclass
@dataclass(frozen=True)
class ImprovementProposal:
    title:str; rationale:str; risk:str="medium"
def propose(title:str,rationale:str,risk:str="medium")->ImprovementProposal:
    if risk not in {"low","medium","high","critical"}: raise ValueError("invalid risk")
    return ImprovementProposal(title,rationale,risk)
