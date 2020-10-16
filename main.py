# referenced URL: https://qiita.com/landmark-b/items/bfbd2d63addd490d58ed

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Conv2D, MaxPooling2D, Flatten
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt

SEED = 0


def main():
    (X_train, y_train), (X_test, y_test) = cifar10.load_data()

    # One-hot encoding
    y_train_ctg = to_categorical(y_train)
    y_test_ctg = to_categorical(y_test)

    # 検証用データの作成
    x_trn, x_val, y_trn, y_val = train_test_split(X_train, y_train_ctg, random_state=SEED)
    print(f'shape of x_trn:{x_trn.shape}, y_trn:{y_trn.shape},x_val:{x_val.shape}, y_val:{y_val.shape}')

    # Inputクラスのインスタンスで入力層(サイズ：(32,32,3))を定義します。
    inputs = Input(shape=(32, 32, 3))

    # 入力層から先のレイヤを追加していきます。
    x = Conv2D(32, (3, 3), activation='relu')(inputs)
    x = Conv2D(32, (3, 3), activation='relu')(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)

    x = Conv2D(32, (3, 3), activation='relu')(x)
    x = Conv2D(32, (3, 3), activation='relu')(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)

    x = Flatten()(x)
    x = Dense(255, activation='relu')(x)
    x = Dense(10, activation='softmax')(x)

    # Modelクラスで入力層と出力層を接続します。
    model_functional = Model(inputs=inputs, outputs=x)

    opt = Adam(lr=0.0001)
    metric_list = ['accuracy']
    model_functional.compile(loss='categorical_crossentropy', optimizer=opt, metrics=metric_list)
    model_functional.summary()

    history_functional = model_functional.fit(x_trn, y_trn, epochs=20, validation_data=(x_val, y_val), batch_size=32)
    show_history(history_functional)


def show_history(history):
    fig, ax = plt.subplots(1, 2, figsize=(15, 5))
    ax[0].set_title('loss')
    #     ax[0].plot(history.epoch, history.history["loss"], label="Train loss")
    ax[0].plot(history.epoch, history.history["val_loss"], label="Validation loss")
    ax[1].set_title('categorical_accuracy')
    #     ax[1].plot(history.epoch, history.history["acc"], label="Train accuracy")
    ax[1].plot(history.epoch, history.history["val_acc"], label="Validation accuracy")
    ax[0].legend()
    ax[1].legend()
    plt.show()



if __name__ == '__main__':
    main()
