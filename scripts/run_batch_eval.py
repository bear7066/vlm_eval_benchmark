import sys
from pathlib import Path
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root / "src"))
from vlm_eval.cli import batch_main


def main():

    args = [
        "--datasets",
        "climbing_ladder",
        "face_planting",
        "falling_off_bike",
        "falling_off_chair",

        "--model_ids",
        "google/gemma-4-E2B-it",
        "bear7011/gemma4-e2b-webvid4K_FT",
        "google/gemma-4-E4B-it",
        "bear7011/gemma4-e4b-webvid4K_FT",

        "--num_frames", "8",
        "--judge_model", "gpt-4o",
    ]

    datasets = args[args.index("--datasets") + 1 : args.index("--model_ids")]
    models = args[args.index("--model_ids") + 1 : args.index("--num_frames")]
    print("=== 開始批次測試任務 ===")
    print(f"資料集: {datasets}")
    print(f"測試模型: {models}")
    print("=" * 25)

    try:
        exit_code = batch_main(args)
        if exit_code == 0:
            print("\n[成功] 所有批次任務已完成。")
        else:
            print(f"\n[錯誤] 任務中斷，錯誤碼: {exit_code}")
        sys.exit(exit_code)
    except Exception as e:
        print(f"\n[異常] 執行過程中發生錯誤: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
