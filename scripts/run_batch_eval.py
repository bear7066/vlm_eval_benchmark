import sys
from pathlib import Path
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root / "src"))
from vlm_eval.cli import batch_main


def main():
    
    args = [
        "--video_dirs", 
        "climbing_ladder", 
        "face_planting", 
        "falling_off_bike", 
        "falling_off_chair",
        
        "--model_ids", 
        "google/gemma-4-E2B-it", 
        "bear7011/gemma4-e2b-webvid4K_FT", 
        "google/gemma-4-E4B-it", 
        "bear7011/gemma4-e4b-webvid4K_FT",
        
        "--dataset_root", "./dataset",
        "--sample_fps", "5.0",           
        "--judge_model", "gpt-4o",
    ]

    print("=== 開始批次測試任務 ===")
    print(f"影片目錄: {args[args.index('--video_dirs')+1 : args.index('--model_ids')]}")
    print(f"測試模型: {args[args.index('--model_ids')+1 : args.index('--dataset_root')]}")
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
