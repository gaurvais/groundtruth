import time
import json
import boto3
from typing import Optional
from botocore.exceptions import ClientError
from .bedrock import InterdictionReport, _VERIFICATION_PROMPT, _crop_stac_visual, _crop_visual_image, log
from utils.config import BEDROCK_MODEL_ID, AWS_REGION

def call_bedrock_verification(
    cluster: dict,
    ndvi_data: dict,
    parcel_id: str = "FARIDABAD-001",
    model_id: str | None = None,
    region: str | None = None,
) -> InterdictionReport:
    """
    Crop the T2 (Post-Disturbance) visual around *cluster*, send to Claude via Bedrock converse,
    validate the response against InterdictionReport schema, and return it.
    """
    target_model_id = model_id or BEDROCK_MODEL_ID
    target_region = region or AWS_REGION

    if "scene_t1" in ndvi_data:
        image_bytes = _crop_stac_visual(cluster, ndvi_data, "scene_t1")
    else:
        image_bytes = _crop_visual_image(
            cluster["bbox"], ndvi_data["shape"], "visual"
        )

    dominant_class = cluster.get("dominant_class", "Agriculture Cropland")
    near_water = cluster.get("near_water", False)
    t0_date = ndvi_data.get("t0_date", "baseline")
    t1_date = ndvi_data.get("t1_date", "post-disturbance")
    
    prompt_text = _VERIFICATION_PROMPT.format(
        lat=cluster["lat"],
        lon=cluster["lon"],
        ndvi_drop=cluster["ndvi_drop_mean"],
        dominant_class=dominant_class,
        near_water=near_water,
        parcel_id=parcel_id,
        t0_date=t0_date,
        t1_date=t1_date,
    )

    client = boto3.client("bedrock-runtime", region_name=target_region)
    max_retries = 3
    base_delay = 2

    for attempt in range(max_retries):
        try:
            response = client.converse(
                modelId=target_model_id,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "image": {
                                    "format": "jpeg",
                                    "source": {"bytes": image_bytes},
                                }
                            },
                            {"text": prompt_text},
                        ],
                    }
                ],
                inferenceConfig={"temperature": 0.0, "maxTokens": 600},
            )

            raw_text: str = (
                response.get("output", {})
                .get("message", {})
                .get("content", [{}])[0]
                .get("text", "")
            )

            clean = raw_text.strip()
            if clean.startswith("```"):
                clean = clean.split("```")[1]
                if clean.startswith("json"):
                    clean = clean[4:]

            report_dict = json.loads(clean)
            report_dict["parcel_id"] = parcel_id
            report = InterdictionReport(**report_dict)
            
            # Attach raw model details for the UI
            report.model_used = target_model_id
            report.region_used = target_region
            report.raw_json = clean
            
            return report

        except ClientError as e:
            error_code = e.response.get("Error", {}).get("Code", "Unknown")
            error_msg = e.response.get("Error", {}).get("Message", str(e))
            log.warning(f"Bedrock ClientError ({error_code}): {error_msg}")
            
            if error_code in ["ThrottlingException", "ModelTimeoutException"]:
                if attempt < max_retries - 1:
                    time.sleep(base_delay * (2 ** attempt))
                    continue
            raise RuntimeError(f"Bedrock API Error [{error_code}]: {error_msg}")
            
        except json.JSONDecodeError as e:
            log.warning(f"Bedrock returned malformed JSON: {e}")
            if attempt < max_retries - 1:
                time.sleep(base_delay)
                continue
            raise RuntimeError(f"Bedrock returned malformed JSON after {max_retries} attempts.")
        except Exception as e:
            raise RuntimeError(f"Unexpected Bedrock error: {str(e)}")

    raise RuntimeError("Max retries exceeded calling Bedrock.")
