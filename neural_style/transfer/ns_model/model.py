from tensorflow.python.keras.models import load_model

from tensorflow.python.keras.layers import Input,  Lambda, Conv2DTranspose
from tensorflow.python.keras.applications.vgg16 import VGG16
from tensorflow.python.keras.models import Model
from tensorflow.python.keras.layers import Conv2D,  BatchNormalization, Add, Activation
from tensorflow.python.keras.preprocessing.image import load_img, img_to_array, array_to_img
import numpy as np

def residual_block(input_ts):
    """ResidualBlockの構築する関数"""
    x = Conv2D(
        128, (3, 3), strides=1, padding='same'
    )(input_ts)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Conv2D(128, (3, 3), strides=1, padding='same')(x)
    x = BatchNormalization()(x)
    return Add()([x, input_ts])



def build_encoder_decoder(input_shape=(224, 224, 3)):
    """変換用ネットワークの構築"""

    # Encoder部分
    input_ts = Input(shape=input_shape, name='input')

    # 入力を[0, 1]の範囲に正規化
    x = Lambda(lambda a: a/255.)(input_ts)

    x = Conv2D(32, (9, 9), strides=1, padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    x = Conv2D(64, (3, 3), strides=2, padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    x = Conv2D(128, (3, 3), strides=2, padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    # ResidualBlockを5ブロック追加
    for _ in range(5):
        x = residual_block(x)

    # Decoder部分
    x = Conv2DTranspose(
            64, (3, 3), strides=2, padding='same'
    )(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    x = Conv2DTranspose(32, (3, 3), strides=2, padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    x = Conv2DTranspose(3, (9, 9), strides=1, padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('tanh')(x)

    # 出力値が[0, 255]になるようにスケール変換
    gen_out = Lambda(lambda a: (a + 1)*127.5)(x)

    model_gen = Model(
        inputs=[input_ts],
        outputs=[gen_out]
    )

    return model_gen


# VGG16のための入力値を正規化する関数
def norm_vgg16(x):
    """RGB->BGR変換と近似的に中心化をおこなう関数"""
    return (x[:, :, :, ::-1] - 120) / 255.


def get_model():
    input_shape = (224, 224, 3)

    # 変換ネットワークの構築
    model_gen = build_encoder_decoder(
        input_shape=input_shape
    )


    # 学習済みモデルVGG16の呼び出し
    vgg16 = VGG16()

    # 重みパラメータを学習させない設定をする
    for layer in vgg16.layers:
        layer.trainable = False



    # 特徴量を抽出する層の名前を定義
    style_layer_names = (
        'block1_conv2',
        'block2_conv2',
        'block3_conv3',
        'block4_conv3'
    )
    contents_layer_names = ('block3_conv3',)

    # 中間層の出力を保持するためのリスト
    style_outputs_gen = []
    contents_outputs_gen = []

    input_gen = model_gen.output  # 変換ネットワークの出力を入力とする
    z = Lambda(norm_vgg16)(input_gen)  # 入力値の正規化
    for layer in vgg16.layers:
        z = layer(z)  # VGG16の層を積み上げてネットワークを再構築
        if layer.name in style_layer_names:
            # スタイル特徴量抽出用の中間層の出力を追加
            style_outputs_gen.append(z)
        if layer.name in contents_layer_names:
            # コンテンツ特徴量抽出用の中間層の出力を追加
            contents_outputs_gen.append(z)

    # モデルを定義
    model = Model(
        inputs=model_gen.input,
        outputs=style_outputs_gen + contents_outputs_gen
    )
    return model_gen, model


"""
model.load_weights("mirror.h5")
trans_img = model_gen.predict(np.expand_dims(img_to_array(load_img("test.jpg", target_size=input_shape[:2])), axis=0))
array_to_img(trans_img[0]).save("trans.jpg")
"""
