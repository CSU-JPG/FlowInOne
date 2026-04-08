"""
VPBench Evaluation Script
HuggingFace: https://huggingface.co/datasets/CSU-JPG/VPBench
Project:     https://csu-jpg.github.io/FlowInOne.github.io/
"""

import argparse
import base64
import io
import json
import logging
import os
import re
import time
from pathlib import Path
from typing import Optional

from datasets import load_dataset
from openai import OpenAI
from PIL import Image

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

# ── Constants ────────────────────────────────────────────────────────────────

DATASET_REPO = "CSU-JPG/VPBench"

ALL_SUBSETS = [
    "class2image",
    "doodles",
    "force",
    "text2image",
    "text_box_control",
    "text_in_image",
    "trajectory",
    "vismarker",
]

# Subsets where the input is a plain text/class canvas → Scenario A
SCENARIO_A_SUBSETS = {"class2image", "text2image"}

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}


# ── Image helpers ─────────────────────────────────────────────────────────────

def pil_to_base64(image: Image.Image) -> str:
    """Convert a PIL Image to a base64-encoded JPEG string."""
    buf = io.BytesIO()
    image.convert("RGB").save(buf, format="JPEG", quality=95)
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def file_to_base64(path: str) -> str:
    """Read an image file and return its base64-encoded string."""
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def find_generated_image(image_name: str, search_root: str) -> Optional[str]:
    """
    Recursively search `search_root` for a file whose stem matches `image_name`
    (comparison is case-insensitive, extension-agnostic).
    Returns the full path of the first match, or None.
    """
    target_stem = Path(image_name).stem.lower()
    for root, _, files in os.walk(search_root):
        for fname in files:
            fpath = Path(fname)
            if fpath.suffix.lower() in IMAGE_EXTS and fpath.stem.lower() == target_stem:
                return os.path.join(root, fname)
    return None


# ── Prompt builders ───────────────────────────────────────────────────────────

def build_scenario_a_prompt(instruction_text: str, b64_source: str, b64_generated: str):
    """Scenario A: text-only / class-to-image generation."""
    return [
        {
            "type": "text",
            "text": (
                "# Role\n"
                "You are an expert evaluator with keen insight in computer vision and image generation. "
                "Your task is to assess the quality of a **text-to-image generation** task.\n\n"
                "# Task Type\n"
                "This is **Scenario A (Text-only / Class-to-Image Generation)**. "
                "The source image is a simple background containing only text or a class label. "
                "You should **ignore the source image** and focus entirely on evaluating whether "
                "the generated image matches the instruction.\n\n"
                "# Inputs\n"
                "1. **Source Image** (for reference only, please ignore it during evaluation):"
            ),
        },
        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64_source}", "detail": "high"}},
        {
            "type": "text",
            "text": (
                f"\n2. **Generation Instruction**: {instruction_text}\n\n"
                "3. **Output Image** (the generated result to evaluate):"
            ),
        },
        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64_generated}", "detail": "high"}},
        {
            "type": "text",
            "text": """
# Evaluation Criteria (Scenario A - Text-to-Image)

1. **Instruction Fidelity**
   - **Goal**: Semantic response accuracy of the generated result to the generation instruction.
   - **Checkpoints**:
     - Do the core objects, attributes (colors, materials, species), and actions described in the instruction accurately appear in the generated image?
     - **Key Determination**: If the generated content is unrelated to the text description, this dimension should receive a low score directly.

2. **Content Consistency**
   - **Goal**: **Canvas Cleanliness**.
   - **Checkpoints**: Is the generated subject clear? Is the background clean or logical? (i.e., should not produce irrelevant hallucinated objects). As long as the generated image is not chaotic, this dimension can receive a high score.

3. **Visual Realism**
   - **Goal**: The naturalness of the image and effective suppression of artifacts.
   - **Checkpoints**: Are there obvious artifacts, blur, edge jaggedness, or limb distortions?

4. **Spatial Control Precision**
   - **Goal**: Compositional Reasonability.
   - **Checkpoints**: Default to **5 points** as long as the object is complete and within the frame.

# Scoring & Output Format
- The `analysis` field must be **concise**: Describe whether the generation follows the instruction and the visual quality.
- Scoring uses a 1-5 point scale.

Please output the evaluation result in JSON format, strictly following the format below (do not add any other text):

{
  "analysis": "Brief analysis.",
  "fidelity_score": 0,
  "consistency_score": 0,
  "realism_score": 0,
  "spatial_score": 0
}""",
        },
    ]


def build_scenario_b_prompt(instruction_text: str, b64_source: str, b64_generated: str):
    """Scenario B: annotated real-world image editing."""
    return [
        {
            "type": "text",
            "text": (
                "# Role\n"
                "You are an expert evaluator with keen insight in computer vision and image generation. "
                "Your task is to assess the quality of a **vision-instruction-based image editing** task.\n\n"
                "# Task Type\n"
                "This is **Scenario B (Annotated Real-world Image Editing)**. "
                "The source image is a natural photograph overlaid with text instructions or visual markers "
                "(such as Bounding Boxes, arrows, scribbles, highlights, etc.). "
                "The model should edit the image according to the instruction while preserving non-edited areas.\n\n"
                "Note: The overlaid text in the source image is the generation instruction. "
                "Since text recognition can be confusing, the text content has been pre-extracted as the "
                "generation instruction below. When evaluating consistency, please ignore the disappearance "
                "of the overlaid instruction text itself and focus on the coherence of image content.\n\n"
                "# Inputs\n"
                "1. **Source Image** (a real-world image with overlaid instructions/visual markers):"
            ),
        },
        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64_source}", "detail": "high"}},
        {
            "type": "text",
            "text": (
                f"\n2. **Generation Instruction**: {instruction_text}\n\n"
                "3. **Output Image** (the generated result to evaluate):"
            ),
        },
        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64_generated}", "detail": "high"}},
        {
            "type": "text",
            "text": """
# Evaluation Criteria (Scenario B - Image Editing)

1. **Instruction Fidelity**
   - **Goal**: Semantic response accuracy of the generated result to the generation instruction.
   - **Checkpoints**:
     - Do the core objects, attributes (colors, materials), and actions described in the instruction accurately appear in the generated image?
     - **Key Determination**: If the generated content is unrelated to the text description, this dimension should receive a low score directly.

2. **Content Consistency**
   - **Goal**: **Background and Non-edited Area Preservation + Instruction/Marker Removal**.
   - **Checkpoints**:
     - Except for areas required to be modified by instructions and markers, the background, lighting, and irrelevant objects must be preserved at pixel-level (or highly similar). **Strictly prohibit** modifying unmarked areas.
     - **CRITICAL**: The source image may contain three types of overlaid elements: (1) text instructions, (2) visual markers (bounding boxes, arrows, scribbles, highlights, etc.), and (3) original background text that is part of the scene. The generated image **MUST strictly remove** all instruction text and visual markers (types 1 & 2). Original background text (type 3) is irrelevant and can be ignored. If instruction text or visual markers still remain in the generated image, this dimension should receive a **low score**.

3. **Visual Realism**
   - **Goal**: The naturalness of the image and effective suppression of artifacts.
   - **Checkpoints**:
     - Are there obvious artifacts, blur, edge jaggedness, or limb distortions?
     - Is the blending between the edited area and the original background natural?

4. **Spatial Control Precision**
   - **Goal**: **Marker Alignment**.
   - **Checkpoints**: Is the generated content strictly confined within the visual markers (boxes, scribbles, etc.)? Is there "overflow" or "underfill"? Does the object follow the direction indicated by arrows? Are the objects indicated by the arrows correctly edited?

# Scoring & Output Format
- The `analysis` field must be **concise**: Describe whether it follows the instruction, visual quality, whether non-edited areas maintain consistency, and whether spatial control is precise.
- Scoring uses a 1-5 point scale.

Please output the evaluation result in JSON format, strictly following the format below (do not add any other text):

{
  "analysis": "Brief analysis.",
  "fidelity_score": 0,
  "consistency_score": 0,
  "realism_score": 0,
  "spatial_score": 0
}""",
        },
    ]


# ── VLM call ──────────────────────────────────────────────────────────────────

def call_vlm(
    client: OpenAI,
    model: str,
    content_parts: list,
    retry_attempts: int = 10,
) -> Optional[str]:
    """Call the VLM API with automatic retry on transient errors."""
    for attempt in range(1, retry_attempts + 1):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": content_parts}],
                temperature=0.2,
                max_tokens=4096,
            )
            return response.choices[0].message.content
        except Exception as e:
            err = str(e)
            logger.warning(f"Attempt {attempt}/{retry_attempts} failed: {e}")
            if "The server had an error" in err:
                time.sleep(30)
            elif "Please try again in " in err:
                try:
                    wait = float(err.split("Please try again in ")[1].split("s.")[0])
                except Exception:
                    wait = 15
                time.sleep(wait * 2)
            else:
                time.sleep(15)
    return None


# ── Result parsing ────────────────────────────────────────────────────────────

def parse_result(response_text: str) -> Optional[dict]:
    """Extract JSON evaluation result from the model response."""
    try:
        return json.loads(response_text)
    except Exception:
        pass
    match = re.search(r"\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}", response_text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(0))
        except Exception:
            pass
    return None


def calculate_verdict(evaluation: dict) -> str:
    """PASS if all four scores > 3, otherwise FAIL."""
    scores = [
        evaluation.get("fidelity_score", 0),
        evaluation.get("consistency_score", 0),
        evaluation.get("realism_score", 0),
        evaluation.get("spatial_score", 0),
    ]
    return "PASS" if all(s > 3 for s in scores) else "FAIL"


# ── Summary report ────────────────────────────────────────────────────────────

def generate_summary(results: list, output_dir: str, run_name: str):
    """Write a Markdown summary report and print subset-level success ratios."""
    if not results:
        return

    total = len(results)
    pass_count = sum(
        1 for r in results
        if r.get("evaluation") and r["evaluation"].get("final_verdict") == "PASS"
    )

    def avg_scores(subset_results):
        keys = ["fidelity_score", "consistency_score", "realism_score", "spatial_score"]
        sums = {k: 0.0 for k in keys}
        valid = 0
        for r in subset_results:
            ev = r.get("evaluation") or {}
            if all(k in ev for k in keys):
                for k in keys:
                    sums[k] += ev[k]
                valid += 1
        if valid:
            return {k: sums[k] / valid for k in keys}, valid
        return {k: 0.0 for k in keys}, 0

    # Per-subset stats
    subsets_in_results = sorted({r["subset"] for r in results})
    subset_lines = ""
    subset_pass_rates = {}
    for s in subsets_in_results:
        sub = [r for r in results if r["subset"] == s]
        s_pass = sum(1 for r in sub if r.get("evaluation") and r["evaluation"].get("final_verdict") == "PASS")
        rate = s_pass / len(sub) if sub else 0.0
        subset_pass_rates[s] = rate
        subset_lines += f"| {s} | {len(sub)} | {s_pass} | {rate:.3f} |\n"

    avgs, valid = avg_scores(results)
    scenario_a = [r for r in results if r["scenario"] == "A"]
    scenario_b = [r for r in results if r["scenario"] == "B"]
    a_avgs, a_valid = avg_scores(scenario_a)
    b_avgs, b_valid = avg_scores(scenario_b)

    report = f"""# VPBench Evaluation Report — {run_name}

## Overall

| Metric | Value |
|--------|-------|
| Total samples | {total} |
| PASS | {pass_count} ({pass_count / total * 100:.1f}%) |
| FAIL | {total - pass_count} ({(total - pass_count) / total * 100:.1f}%) |
| Valid evaluations | {valid} |

## Success Ratio by Subset

| Subset | Samples | PASS | Success Ratio |
|--------|---------|------|---------------|
{subset_lines}
## Average Scores (1–5 scale)

|  | Fidelity | Consistency | Realism | Spatial |
|--|----------|-------------|---------|---------|
| **Overall** ({valid}) | {avgs['fidelity_score']:.2f} | {avgs['consistency_score']:.2f} | {avgs['realism_score']:.2f} | {avgs['spatial_score']:.2f} |
| **Scenario A** – text/class-to-image ({a_valid}) | {a_avgs['fidelity_score']:.2f} | {a_avgs['consistency_score']:.2f} | {a_avgs['realism_score']:.2f} | {a_avgs['spatial_score']:.2f} |
| **Scenario B** – image editing ({b_valid}) | {b_avgs['fidelity_score']:.2f} | {b_avgs['consistency_score']:.2f} | {b_avgs['realism_score']:.2f} | {b_avgs['spatial_score']:.2f} |

## Per-Sample Details

"""
    for r in results:
        ev = r.get("evaluation") or {}
        report += f"### [{r['subset']}] {r['image_name']} (Scenario {r['scenario']})\n"
        report += f"- Instruction: {r['instruction']}\n"
        report += f"- Generated image: {r.get('generated_image', 'N/A')}\n"
        if ev:
            report += (
                f"- Verdict: **{ev.get('final_verdict', 'N/A')}**  "
                f"Fidelity={ev.get('fidelity_score','?')} "
                f"Consistency={ev.get('consistency_score','?')} "
                f"Realism={ev.get('realism_score','?')} "
                f"Spatial={ev.get('spatial_score','?')}\n"
            )
            report += f"- Analysis: {ev.get('analysis', '')}\n"
        else:
            report += "- Evaluation failed.\n"
        report += "\n"

    report_path = os.path.join(output_dir, f"summary_{run_name}.md")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report)
    logger.info(f"Summary report saved to: {report_path}")

    # Print success ratios
    print("\n" + "=" * 50)
    print(f"VPBench Results — {run_name}")
    print("=" * 50)
    header = f"{'Subset':<22} {'N':>5} {'PASS':>5} {'Rate':>7}"
    print(header)
    print("-" * len(header))
    for s in subsets_in_results:
        sub = [r for r in results if r["subset"] == s]
        s_pass = sum(1 for r in sub if r.get("evaluation") and r["evaluation"].get("final_verdict") == "PASS")
        print(f"{s:<22} {len(sub):>5} {s_pass:>5} {subset_pass_rates[s]:>7.3f}")
    print("-" * len(header))
    print(f"{'Total':<22} {total:>5} {pass_count:>5} {pass_count / total:>7.3f}")
    print("=" * 50 + "\n")


# ── Main evaluation loop ──────────────────────────────────────────────────────

def evaluate_subset(
    client: OpenAI,
    model: str,
    subset: str,
    generated_dir: str,
    output_dir: str,
    run_name: str,
    max_samples: Optional[int],
) -> list:
    """Evaluate one VPBench subset. Returns a list of result dicts."""
    logger.info(f"Loading subset '{subset}' from {DATASET_REPO} ...")
    ds = load_dataset(DATASET_REPO, name=subset, split="train")
    logger.info(f"  {len(ds)} samples loaded.")

    results = []
    skipped = 0
    scenario_stats = {"A": 0, "B": 0}

    for idx, sample in enumerate(ds):
        if max_samples and len(results) >= max_samples:
            break

        image_name      = sample["image_name"]
        instruction     = sample.get("recognized_text") or ""
        source_pil      = sample["input_image"]          # PIL.Image
        scenario        = "A" if subset in SCENARIO_A_SUBSETS else "B"

        # Find the user's generated image
        gen_path = find_generated_image(image_name, generated_dir)
        if gen_path is None:
            logger.warning(f"  [SKIP] No generated image found for: {image_name}")
            skipped += 1
            continue

        scenario_stats[scenario] += 1
        logger.info(
            f"  [{len(results) + 1}] [Scenario {scenario}] {image_name} | "
            f"instruction: {instruction[:60]}..."
        )

        b64_source    = pil_to_base64(source_pil)
        b64_generated = file_to_base64(gen_path)

        if scenario == "A":
            content = build_scenario_a_prompt(instruction, b64_source, b64_generated)
        else:
            content = build_scenario_b_prompt(instruction, b64_source, b64_generated)

        response_text = call_vlm(client, model, content)

        evaluation = None
        if response_text:
            evaluation = parse_result(response_text)
            if evaluation:
                evaluation["final_verdict"] = calculate_verdict(evaluation)
                avg = sum([
                    evaluation.get("fidelity_score", 0),
                    evaluation.get("consistency_score", 0),
                    evaluation.get("realism_score", 0),
                    evaluation.get("spatial_score", 0),
                ]) / 4
                logger.info(
                    f"    → {evaluation['final_verdict']} "
                    f"(avg={avg:.1f}  F={evaluation.get('fidelity_score')} "
                    f"C={evaluation.get('consistency_score')} "
                    f"R={evaluation.get('realism_score')} "
                    f"S={evaluation.get('spatial_score')})"
                )
            else:
                logger.warning("    → Failed to parse evaluation JSON.")
        else:
            logger.error("    → API call failed after all retries.")

        results.append({
            "index":           idx,
            "subset":          subset,
            "scenario":        scenario,
            "image_name":      image_name,
            "generated_image": gen_path,
            "instruction":     instruction,
            "evaluation":      evaluation,
            "raw_response":    response_text,
        })

        # Checkpoint save every 10 samples
        if len(results) % 10 == 0:
            _save_results(results, output_dir, run_name, subset)

    logger.info(
        f"  Subset '{subset}' done: {len(results)} evaluated, {skipped} skipped. "
        f"Scenario A={scenario_stats['A']} B={scenario_stats['B']}"
    )
    return results


def _save_results(results: list, output_dir: str, run_name: str, subset: str):
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, f"results_{run_name}_{subset}.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    logger.info(f"  Checkpoint saved → {path}")


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="VPBench evaluation script (loads benchmark via load_dataset)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--generated_dir", required=True,
        help="Root directory containing your model's generated images.",
    )
    parser.add_argument(
        "--output_dir", default="./vpbench_results",
        help="Directory to save evaluation results (default: ./vpbench_results).",
    )
    parser.add_argument(
        "--run_name", default="eval",
        help="Name tag appended to output files (default: eval).",
    )
    parser.add_argument(
        "--model", default="gpt-5.2",
        help="OpenAI-compatible model name (default: gpt-5.2).",
    )
    parser.add_argument(
        "--api_key", default=None,
        help="OpenAI API key. Falls back to OPENAI_API_KEY environment variable.",
    )
    parser.add_argument(
        "--base_url", default=None,
        help="Custom API base URL (optional, e.g. for proxies). "
             "Defaults to the official OpenAI endpoint.",
    )
    parser.add_argument(
        "--subsets", nargs="+", default=ALL_SUBSETS,
        choices=ALL_SUBSETS,
        help="Subsets to evaluate (default: all 8 subsets).",
    )
    parser.add_argument(
        "--max_samples", type=int, default=None,
        help="Max samples per subset (default: all).",
    )
    args = parser.parse_args()

    # Resolve API key
    api_key = args.api_key or os.environ.get("OPENAI_API_KEY")
    if not api_key:
        parser.error(
            "An OpenAI API key is required. Provide it via --api_key or "
            "set the OPENAI_API_KEY environment variable."
        )

    os.makedirs(args.output_dir, exist_ok=True)

    client_kwargs = {"api_key": api_key}
    if args.base_url:
        client_kwargs["base_url"] = args.base_url
    client = OpenAI(**client_kwargs)

    logger.info(f"Model:         {args.model}")
    logger.info(f"Subsets:       {args.subsets}")
    logger.info(f"Generated dir: {args.generated_dir}")
    logger.info(f"Output dir:    {args.output_dir}")

    all_results = []
    for subset in args.subsets:
        subset_results = evaluate_subset(
            client=client,
            model=args.model,
            subset=subset,
            generated_dir=args.generated_dir,
            output_dir=args.output_dir,
            run_name=args.run_name,
            max_samples=args.max_samples,
        )
        all_results.extend(subset_results)
        _save_results(subset_results, args.output_dir, args.run_name, subset)

    # Save combined results
    combined_path = os.path.join(args.output_dir, f"results_{args.run_name}_all.json")
    with open(combined_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)
    logger.info(f"All results saved → {combined_path}")

    generate_summary(all_results, args.output_dir, args.run_name)


if __name__ == "__main__":
    main()