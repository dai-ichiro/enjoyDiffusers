import os
import sys
import cv2
import numpy as np
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QSlider, QLabel, QFileDialog, QMessageBox, QFrame,
    QSplitter, QScrollArea
)
from PySide6.QtCore import Qt, QTimer, Signal, Slot, QSize
from PySide6.QtGui import QImage, QPixmap, QPainter, QPen, QColor


class ImageWidget(QWidget):
    """OpenCVの画像を表示し、マウスイベントを処理するウィジェット"""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.original_image = None
        self.display_image = None
        self.mask_image = None
        self.history = []  # マスク履歴を保存するリスト
        self.drawing = False
        self.rectangle_size = 20
        self.last_x, self.last_y = -1, -1
        self.scale_factor = 1.0  # 表示スケール
        
        # マウストラッキングを有効化
        self.setMouseTracking(True)
        # ウィジェットがフォーカスを受け取れるようにする
        self.setFocusPolicy(Qt.StrongFocus)
        
        # 初期サイズを800x800に設定
        self.setMinimumSize(800, 800)
    
    def load_image(self, image_path):
        """画像を読み込み、表示用の準備をする"""
        try:
            self.original_image = cv2.imread(image_path)
            if self.original_image is None:
                raise FileNotFoundError(f"Could not open or find the image: {image_path}")
            
            # 表示用のコピーを作成
            self.display_image = self.original_image.copy()
            
            # マスク画像を初期化
            self.mask_image = np.zeros(self.original_image.shape[:2], dtype=np.uint8)
            
            # 履歴をクリア
            self.history = [self.mask_image.copy()]
            
            # 画像のスケールを計算
            self.calculate_scale_factor()
            
            # 画像を表示
            self.update()
            
            return True
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to load image: {str(e)}")
            return False
    
    def calculate_scale_factor(self):
        """画像が800x800に収まるようにスケール係数を計算"""
        if self.original_image is None:
            return
        
        h, w = self.original_image.shape[:2]
        max_dim = 800
        
        if w > max_dim or h > max_dim:
            self.scale_factor = min(max_dim / w, max_dim / h)
        else:
            self.scale_factor = 1.0
        
        # ウィジェットのサイズを計算されたスケールに基づいて設定
        scaled_w = int(w * self.scale_factor)
        scaled_h = int(h * self.scale_factor)
        self.setMinimumSize(scaled_w, scaled_h)
        self.setMaximumSize(scaled_w, scaled_h)
    
    def set_rectangle_size(self, size):
        """四角形の大きさを設定"""
        self.rectangle_size = size
    
    def save_mask(self, filename):
        """マスク画像を保存"""
        if self.mask_image is not None:
            try:
                cv2.imwrite(filename, self.mask_image)
                return True
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Failed to save mask: {str(e)}")
        else:
            QMessageBox.warning(self, "Warning", "No mask to save")
        return False
    
    def undo(self):
        """マスクの変更を元に戻す"""
        if len(self.history) > 1:
            # 最後の状態を削除
            self.history.pop()
            # 前の状態を現在のマスクとして設定
            self.mask_image = self.history[-1].copy()
            # 表示を更新
            self.update_display()
            return True
        else:
            QMessageBox.information(self, "情報", "これ以上元に戻せません")
            return False
    
    def update_display(self):
        """表示画像を更新（マスクを表示）"""
        if self.original_image is None or self.mask_image is None:
            return
        
        # 元画像のコピーを取得
        self.display_image = self.original_image.copy()
        
        # マスク部分を白色で塗りつぶす（不透明）
        self.display_image[self.mask_image > 0] = [255, 255, 255]  # 白色（BGR形式）
        
        # 表示を更新
        self.update()
    
    def paintEvent(self, event):
        """画像を描画"""
        if self.display_image is not None:
            painter = QPainter(self)
            
            # OpenCVのBGR画像をRGBに変換
            rgb_image = cv2.cvtColor(self.display_image, cv2.COLOR_BGR2RGB)
            h, w, ch = rgb_image.shape
            
            # スケールを適用
            scaled_w = int(w * self.scale_factor)
            scaled_h = int(h * self.scale_factor)
            
            # リサイズした画像を作成
            if self.scale_factor != 1.0:
                rgb_image = cv2.resize(rgb_image, (scaled_w, scaled_h), interpolation=cv2.INTER_AREA)
            
            # QImageに変換
            bytes_per_line = 3 * scaled_w
            qt_image = QImage(rgb_image.data, scaled_w, scaled_h, bytes_per_line, QImage.Format_RGB888)
            
            # 画像を描画
            painter.drawImage(0, 0, qt_image)
    
    def mousePressEvent(self, event):
        """マウスボタンが押されたときの処理"""
        if event.button() == Qt.LeftButton and self.display_image is not None:
            self.drawing = True
            # スケール係数を考慮してマウス位置を補正
            x = int(event.position().x() / self.scale_factor)
            y = int(event.position().y() / self.scale_factor)
            self.draw_rectangle(x, y)
    
    def mouseMoveEvent(self, event):
        """マウスが移動したときの処理"""
        if self.drawing and self.display_image is not None:
            # スケール係数を考慮してマウス位置を補正
            x = int(event.position().x() / self.scale_factor)
            y = int(event.position().y() / self.scale_factor)
            self.draw_rectangle(x, y)
    
    def mouseReleaseEvent(self, event):
        """マウスボタンが離されたときの処理"""
        if event.button() == Qt.LeftButton and self.drawing:
            self.drawing = False
            # 現在のマスク状態を履歴に追加
            if self.mask_image is not None:
                self.history.append(self.mask_image.copy())
    
    def draw_rectangle(self, x, y):
        """指定された位置に四角形を描画"""
        if self.display_image is None or self.mask_image is None:
            return
        
        # 四角形の座標を計算
        h, w = self.original_image.shape[:2]
        xmin = max(0, x - self.rectangle_size)
        ymin = max(0, y - self.rectangle_size)
        xmax = min(w - 1, x + self.rectangle_size)
        ymax = min(h - 1, y + self.rectangle_size)
        
        # 四角形をマスクに描画
        cv2.rectangle(self.mask_image, (xmin, ymin), (xmax, ymax), 255, -1)
        
        # 表示を更新
        self.update_display()


class MaskCreatorWindow(QMainWindow):
    """マスク作成アプリケーションのメインウィンドウ"""
    
    def __init__(self):
        super().__init__()
        
        self.image_path = None
        self.init_ui()
    
    def init_ui(self):
        """UIの初期化"""
        # ウィンドウの設定
        self.setWindowTitle("マスク作成ツール")
        self.resize(1200, 900)  # ウィンドウサイズをさらに大きく設定
        
        # 中央ウィジェット
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        # メインレイアウト
        main_layout = QHBoxLayout(central_widget)
        
        # スプリッター（画像エリアとコントロールパネルを分ける）
        splitter = QSplitter(Qt.Horizontal)
        main_layout.addWidget(splitter)
        
        # 画像表示用のスクロールエリア
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        # スクロールエリアのスクロールバーを常に非表示にする
        scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        scroll_area.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        scroll_area.setMinimumSize(820, 820)  # 最小サイズを820x820に設定
        splitter.addWidget(scroll_area)
        
        # 画像表示ウィジェット
        self.image_widget = ImageWidget()
        scroll_area.setWidget(self.image_widget)
        
        # 右側のコントロールパネル
        control_panel = QWidget()
        control_layout = QVBoxLayout(control_panel)
        splitter.addWidget(control_panel)
        
        # スプリッターの初期サイズ比率を設定
        splitter.setSizes([850, 300])
        
        # コントロールパネルの内容
        # ファイル選択ボタン
        self.open_button = QPushButton("画像を開く")
        self.open_button.clicked.connect(self.open_image)
        control_layout.addWidget(self.open_button)
        
        # サイズスライダー
        size_frame = QFrame()
        size_layout = QVBoxLayout(size_frame)
        size_label = QLabel("矩形サイズ:")
        size_layout.addWidget(size_label)
        
        self.size_slider = QSlider(Qt.Horizontal)
        self.size_slider.setMinimum(1)
        self.size_slider.setMaximum(100)
        self.size_slider.setValue(20)
        self.size_slider.valueChanged.connect(self.update_rectangle_size)
        size_layout.addWidget(self.size_slider)
        
        self.size_value_label = QLabel("20")
        size_layout.addWidget(self.size_value_label)
        
        control_layout.addWidget(size_frame)
        
        # 元に戻すボタン
        self.undo_button = QPushButton("元に戻す")
        self.undo_button.clicked.connect(self.undo_action)
        control_layout.addWidget(self.undo_button)
        
        # 保存ボタン
        self.save_button = QPushButton("マスクを保存")
        self.save_button.clicked.connect(self.save_mask)
        control_layout.addWidget(self.save_button)
        
        # 操作説明
        instruction_label = QLabel(
            "使い方:\n"
            "1. 「画像を開く」で画像を選択 または\n"
            "   画像をドラッグ＆ドロップで読み込み\n"
            "2. マウスの左ボタンでドラッグして描画\n"
            "3. スライダーで矩形サイズを調整\n"
            "4. 「元に戻す」で前の状態に戻る\n"
            "5. 「マスクを保存」でマスク画像を保存\n\n"
            "※ マスク領域は白色で表示されます"
        )
        instruction_label.setWordWrap(True)
        control_layout.addWidget(instruction_label)
        
        # スペースを埋めるための伸縮スペーサー
        control_layout.addStretch(1)
        
        # ドラッグ＆ドロップを有効化
        self.setAcceptDrops(True)
    
    def update_rectangle_size(self, value):
        """スライダーの値が変更されたときに呼ばれる"""
        self.size_value_label.setText(str(value))
        self.image_widget.set_rectangle_size(value)
    
    def undo_action(self):
        """元に戻す操作"""
        self.image_widget.undo()
    
    def open_image(self):
        """画像を開くダイアログを表示"""
        file_dialog = QFileDialog()
        image_path, _ = file_dialog.getOpenFileName(
            self, "開く画像を選択", "", "画像ファイル (*.png *.jpg *.jpeg *.bmp)"
        )
        
        if image_path:
            self.image_path = image_path
            if self.image_widget.load_image(image_path):
                self.setWindowTitle(f"マスク作成ツール - {os.path.basename(image_path)}")
    
    def save_mask(self):
        """マスク画像を保存するダイアログを表示"""
        if self.image_path:
            # デフォルトのファイル名を生成
            default_filename = os.path.splitext(os.path.basename(self.image_path))[0] + "_mask.png"
            default_path = os.path.join(os.path.dirname(self.image_path), default_filename)
            
            file_dialog = QFileDialog()
            mask_path, _ = file_dialog.getSaveFileName(
                self, "マスク画像を保存", default_path, "PNG画像 (*.png)"
            )
            
            if mask_path:
                if self.image_widget.save_mask(mask_path):
                    QMessageBox.information(self, "成功", f"マスク画像が保存されました: {mask_path}")
        else:
            QMessageBox.warning(self, "警告", "まず画像を開いてください")
    
    def dragEnterEvent(self, event):
        """ドラッグされたファイルがウィンドウ上に入ってきたときの処理"""
        # URLかファイルの場合のみ受け入れる
        if event.mimeData().hasUrls():
            event.acceptProposedAction()
    
    def dropEvent(self, event):
        """ドロップされたファイルを処理"""
        # ドロップされたファイルのURLを取得
        urls = event.mimeData().urls()
        if urls and len(urls) > 0:
            # 最初のファイルだけ処理
            file_path = urls[0].toLocalFile()
            file_ext = os.path.splitext(file_path)[1].lower()
            
            # 画像ファイルの拡張子かどうかチェック
            if file_ext in ['.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff']:
                self.image_path = file_path
                if self.image_widget.load_image(file_path):
                    self.setWindowTitle(f"マスク作成ツール - {os.path.basename(file_path)}")
            else:
                QMessageBox.warning(self, "警告", "サポートされていないファイル形式です。画像ファイルをドロップしてください。")


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MaskCreatorWindow()
    window.show()
    sys.exit(app.exec())
