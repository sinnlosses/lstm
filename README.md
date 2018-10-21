# lst
m
手順を示す

1. 学習データを用意する
　この学習データは生の文であり、一行に一文が記載されているものとする。
2. lstm_template.pyを起動する
　一番最初はword2vecを読み込むため時間がかかるが、embeddingを保存するので
　2回目以降は速くなる
3. 学習が終わると重みとモデルが保存されている。
4. 品詞の並びをサンプリングするため、predict_sampling.pyを起動する。
5. 生成されたファイルはスペース区切りのサンプルコピー、品詞のリストの順でカンマ区切りとなっている。
6. lstm_in_filling.pyを起動してコピーの結果を見る


keras_tips:
Layerの出力shapeを知りたい
    model.layers[0].get_output_at(0).get_shape().as_list()
