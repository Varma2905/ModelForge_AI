from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from app.hf import client, inference
from app.hf.compatibility import CompatibilityResult, assess_regression_compatibility

# Plain functions, not a formal provider ABC/class hierarchy — there is only
# one provider (Hugging Face) today. app/connectors/base.py's ABC earns its
# abstraction with 3 concrete implementations sharing one call site; a
# single-implementation class here would be premature. Revisit if a second
# model source is ever added.


async def search(query: Optional[str], task: Optional[str] = None, limit: int = 20) -> List[Dict[str, Any]]:
    return await client.search_models(query, task=task, limit=limit)


async def get_details(model_id: str) -> Dict[str, Any]:
    return await client.get_model_info(model_id)


async def check_compatibility(model_id: str, dataset_schema: Dict[str, Any]) -> CompatibilityResult:
    model_info = await client.get_model_info(model_id)
    return assess_regression_compatibility(model_info, dataset_schema)


async def run(model_id: str, filename: str, X: pd.DataFrame) -> Tuple[Any, np.ndarray, Optional[str]]:
    estimator, version_warning = await inference.download_and_load(model_id, filename)
    predictions = inference.run_inference(estimator, X)
    return estimator, predictions, version_warning
