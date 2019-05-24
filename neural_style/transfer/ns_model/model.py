from tensorflow.python.keras.models import load_model

from tensorflow.python.keras.layers import Input,  Lambda, Conv2DTranspose
from tensorflow.python.keras.applications.vgg16 import VGG16
from tensorflow.python.keras.models import Model
from tensorflow.python.keras.layers import Conv2D,  BatchNormalization, Add, Activation
from tensorflow.python.keras.preprocessing.image import load_img, img_to_array, array_to_img
from tensorflow.python.keras import backend as K
from tensorflow.python.keras.optimizers import Adadelta
import numpy as np
import math
import glob
import shutil
import os


class NS_MODEL:
    def __init__(self):
        input_shape = (224, 224, 3)

        # 変換ネットワークの構築
        self.model_gen = self.build_encoder_decoder(
            input_shape=input_shape
        )

        # 学習済みモデルVGG16の呼び出し
        self.vgg16 = VGG16()

        # 重みパラメータを学習させない設定をする
        for layer in self.vgg16.layers:
            layer.trainable = False

        # 特徴量を抽出する層の名前を定義
        self.style_layer_names = (
            'block1_conv2',
            'block2_conv2',
            'block3_conv3',
            'block4_conv3'
        )
        self.contents_layer_names = ('block3_conv3',)

        # 中間層の出力を保持するためのリスト
        self.style_outputs_gen = []
        self.contents_outputs_gen = []

        self.input_gen = self.model_gen.output  # 変換ネットワークの出力を入力とする
        z = Lambda(self.norm_vgg16)(self.input_gen)  # 入力値の正規化
        for layer in self.vgg16.layers:
            z = layer(z)  # VGG16の層を積み上げてネットワークを再構築
            if layer.name in self.style_layer_names:
                # スタイル特徴量抽出用の中間層の出力を追加
                self.style_outputs_gen.append(z)
            if layer.name in self.contents_layer_names:
                # コンテンツ特徴量抽出用の中間層の出力を追加
                self.contents_outputs_gen.append(z)

        # モデルを定義
        self.model = Model(
            inputs=self.model_gen.input,
            outputs=self.style_outputs_gen + self.contents_outputs_gen
        )
        input_sty = Input(shape=input_shape, name='input_sty')

        style_outputs = []
        x = Lambda(self.norm_vgg16)(input_sty)
        for layer in self.vgg16.layers:
            x = layer(x)
            if layer.name in self.style_layer_names:
                style_outputs.append(x)

        self.model_sty = Model(
            inputs=input_sty,
            outputs=style_outputs
        )

        input_con = Input(shape=input_shape, name='input_con')

        contents_outputs = []
        y = Lambda(self.norm_vgg16)(input_con)
        for layer in self.vgg16.layers:
            y = layer(y)
            if layer.name in self.contents_layer_names:
                contents_outputs.append(y)

        self.model_con = Model(
            inputs = input_con,
            outputs = contents_outputs
        )

    def residual_block(self,input_ts):
        """ResidualBlockの構築する関数"""
        x = Conv2D(
            128, (3, 3), strides=1, padding='same'
        )(input_ts)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = Conv2D(128, (3, 3), strides=1, padding='same')(x)
        x = BatchNormalization()(x)
        return Add()([x, input_ts])

    def build_encoder_decoder(self,input_shape=(224, 224, 3)):
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
            x = self.residual_block(x)

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
    def norm_vgg16(self,x):
        """RGB->BGR変換と近似的に中心化をおこなう関数"""
        return (x[:, :, :, ::-1] - 120) / 255.


    # 画像ファイル読み込み用のラッパー関数定義
    def load_imgs(self,img_paths, target_size=(224, 224)):
        """画像ファイルのパスのリストから、配列のバッチを返す"""
        _load_img = lambda x: img_to_array(
            load_img(x, target_size=target_size)
        )
        img_list = [
            np.expand_dims(_load_img(img_path), axis=0)
            for img_path in img_paths
        ]
        return np.concatenate(img_list, axis=0)


    def train_generator(self,img_paths, batch_size, model, y_true_sty, shuffle=True, epochs=None):
        """学習データを生成するジェネレータ"""
        n_samples = len(img_paths)
        indices = list(range(n_samples))
        steps_per_epoch = math.ceil(n_samples / batch_size)
        img_paths = np.array(img_paths)
        cnt_epoch = 0
        while True:
            cnt_epoch += 1
            if shuffle:
                np.random.shuffle(indices)
            for i in range(steps_per_epoch):
                start = batch_size*i
                end = batch_size*(i + 1)
                X = self.load_imgs(img_paths[indices[start:end]])
                batch_size_act = X.shape[0]
                y_true_sty_t = [
                    np.repeat(feat, batch_size_act, axis=0)
                    for feat in y_true_sty
                ]
                # コンテンツ特徴量の抽出
                y_true_con = model.predict(X)
                yield (X,  y_true_sty_t + [y_true_con])
            if epochs is not None:
                if cnt_epoch >= epochs:
                    raise StopIteration


    def feature_loss(self, y_true, y_pred):
        """コンテンツ特徴量の損失関数"""
        norm = K.prod(K.cast(K.shape(y_true)[1:], 'float32'))
        return K.sum(
            K.square(y_pred - y_true), axis=(1, 2, 3)
        )/norm

    def gram_matrix(self, X):
        """グラム行列の算出"""
        X_sw = K.permute_dimensions(
            X, (0, 3, 2, 1)
        )  # 軸の入れ替え
        s = K.shape(X_sw)
        new_shape = (s[0], s[1], s[2]*s[3])
        X_rs = K.reshape(X_sw, new_shape)
        X_rs_t = K.permute_dimensions(
            X_rs, (0, 2, 1)
        )  # 行列の転置
        dot = K.batch_dot(X_rs, X_rs_t)  # 内積の計算
        norm = K.prod(K.cast(s[1:], 'float32'))
        return dot/norm


    def style_loss(self, y_true, y_pred):
        """スタイル用の損失関数定義"""
        return K.sum(
            K.square(
                self.gram_matrix(y_pred) - self.gram_matrix(y_true)
            ),
            axis=(1, 2)
        )

    # Total Variation Regularizerの定義
    def TVRegularizer(self, x, weight=1e-6, beta=1.0, input_size=(224, 224)):
        delta = 1e-8
        h, w = input_size
        d_h = K.square(x[:, :h - 1, :w - 1, :] - x[:, 1:, :w - 1, :])
        d_w = K.square(x[:, :h - 1, :w - 1, :] - x[:, :h - 1, 1:, :])
        return weight * K.mean(K.sum(K.pow(d_h + d_w + delta, beta/2.)))


    def learn(self, style_name):
        img_paths = glob.glob("transfer/ns_model/train_img/*")
        batch_size = 2
        epochs = 5
        input_shape = (224, 224, 3)
        input_size = input_shape[:2]
        style = glob.glob("media/style/*")[0].split("\\")[-1]

        img_sty = load_img(
            'media/style/'+style,
            target_size=input_size
        )
        img_arr_sty = np.expand_dims(img_to_array(img_sty), axis=0)
        self.y_true_sty = self.model_sty.predict(img_arr_sty)
        shutil.rmtree("./media/style")
        os.mkdir("./media/style")

        self.gen = self.train_generator(
            img_paths,
            batch_size,
            self.model_con,
            self.y_true_sty,
            epochs=epochs
        )

        gen_output_layer = self.model_gen.layers[-1]
        tv_loss = self.TVRegularizer(gen_output_layer.output)
        gen_output_layer.add_loss(tv_loss)

        self.model.compile(
            optimizer = Adadelta(),
            loss = [
                self.style_loss,
                self.style_loss,
                self.style_loss,
                self.style_loss,
                self.feature_loss
            ],
            loss_weights = [1.0, 1.0, 1.0, 1.0, 3.0]
        )

        now_epoch = 0
        min_loss = np.inf
        steps_per_epoch = math.ceil(len(img_paths)/batch_size)

        # 学習
        for i , (X_train, y_train) in enumerate(self.gen):
            if i % steps_per_epoch == 0:
                now_epoch += 1

            loss = self.model.train_on_batch(X_train, y_train)
            if loss[0]<min_loss:
                min_loss = loss[0]
                self.model.save("transfer/ns_model/pretrained_model/" + style_name + ".h5")

            print("epoch: {}, iters: {}, loss: {:.3f}".format(now_epoch, i, loss[0]))



