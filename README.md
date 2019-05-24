# neural-style-django

## 利用フレームワーク
Python 3.6.4
* Django (2.1.7)
* Celery (4.2.2)
* Tensorflow (1.13.1)

## 画風変換アプリケーション
ユーザーが選択した画像を，絵画の画風と同じように変換するアプリケーション

___
### 画像の変換#### ホーム画面 

![ホーム画面](https://github.com/natsu-summer72/neural-style/blob/master/example/home.png)


#### 画風変換処理の実行中画面

機械学習の重い処理は非同期で実行
非同期処理は，Django+Celery により実装

![非同期](https://github.com/natsu-summer72/neural-style/blob/master/example/async.png)

#### 結果の出力画面
![結果](https://github.com/natsu-summer72/neural-style/blob/master/example/result.png)

___
### 新しいスタイルの学習
新しいスタイルを学習することもできる。
学習の処理は，Celery Workerが実行

#### Django + Celery による非同期処理
  * 新しいスタイルの学習には，時間がかかってしまいその間 HTTP 通信を継続するのは無駄。
  * 機械学習による重い処理は，Workerが実行し，サーバーはHTTPレスポンスを返す。
![celery](https://github.com/natsu-summer72/neural-style/blob/master/example/celery.png)

例: example という名前のスタイル画像を追加
![スタイルの追加](https://github.com/natsu-summer72/neural-style/blob/master/example/learn_home.png)

新しいスタイルの学習は，Celery Worker が実行
![学習](https://github.com/natsu-summer72/neural-style/blob/master/example/learning.png)

追加したスタイルに変換することが可能になる
![スタイルの追加後](https://github.com/natsu-summer72/neural-style/blob/master/example/finish_learning.png)


