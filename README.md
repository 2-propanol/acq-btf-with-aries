# Aries4軸ステージでBTF撮影
[![Code style: black](https://img.shields.io/badge/code%20style-black-black?style=flat-square)](https://github.com/psf/black)

## 初期設定
### Spinnakerのインストール
[FLIRのページ](https://flir.app.boxcn.net/v/SpinnakerSDK)から環境に合ったSpinnakerダウンロードし、インストールする。
```bash
# example
poetry add spinnaker_python-2.4.0.143-cp38-cp38-win_amd64.whl
```

## 撮影まで
- `poetry run`を頭に付けるか`& ***/activate.ps1`を付けて仮想環境を有効化する
- preview_calib.py
    - なるべく平行にカメラを設置、ピントを合わせる。
    - ロール軸が45度回転しても素材が画面に映るようにする。
    - ARマーカーをステージに設置。
- auto_calib.py
    - `FILENAME_*`を書き換え
    - SpinView でマーカーがいい感じに認識されるようにパラメーターを調整する
- preview_calib.py
    - キャリブレーション出来てるか確認する
    - ホワイトバランス調整
    - Gammaを1にする
    - Gainを0にする
    - 露出設定する(MIN, MAX, REF)
    - ノイズの量を考えてAVERAGEを設定する
        - PowerToysのカラーピッカーを活用する(Win+Shift+C)
- acq_btf.py
    - 保存するファイル名を決める
    - キャリブレーションファイルが合ってるか確認する
    - offset確認
    - 撮影preset(1D359, 2D120, 4D625とか)設定
