from modelscope.hub.snapshot_download import snapshot_download

import config as cfg


def main():
    model_dir = snapshot_download(
        model_id=cfg.MODEL_ID,
        local_dir=str(cfg.MODEL_PATH),
        max_workers=4,
    )
    print('模型下载完成：')
    print(model_dir)


if __name__ == '__main__':
    main()
