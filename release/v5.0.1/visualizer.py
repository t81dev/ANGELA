# visualizer.py  (repurposed, headless)
from typing import Any, Dict, List

def renderBranchTree(branches: List[Dict], selected_id=None) -> Dict[str, Any]:
    return {"type":"BranchTree","payload":[
        {"id":b.get("id"), "score":b.get("score"), "parent":b.get("parent")}
        for b in branches
    ], "selected": selected_id, "version":"1.0"}

def view_trait_resonance(traits: Dict[str, float]) -> Dict[str, Any]:
    return {"type":"TraitResonance","payload":[
        {"trait":k, "amplitude":float(v)} for k,v in traits.items()
    ], "summary":{
        "max": max(traits, key=traits.get) if traits else None,
        "mean": (sum(traits.values())/max(len(traits),1)) if traits else 0.0
    }, "version":"1.0"}
