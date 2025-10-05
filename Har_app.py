# Copyright (c) [2025] [Mathesh Vaithiyanathan]
# This software is licensed under the MIT License.
# See the LICENSE file for details.

import sys
import os
import cv2
import numpy as np
import base64
import json
import ctypes
from PyQt5.QtWidgets import (QApplication, QMainWindow, QLabel, QVBoxLayout,
                             QWidget, QHBoxLayout, QStatusBar, QGraphicsView,
                             QGraphicsScene, QGraphicsPathItem, QLineEdit,
                             QPushButton, QRadioButton, QGraphicsLineItem,
                             QMenu, QAction, QFileDialog, QMessageBox, QTabWidget,
                             QInputDialog, QTableWidget, QTableWidgetItem,
                             QHeaderView, QAbstractItemView, QStyle)
from PyQt5.QtGui import QImage, QPixmap, QPainterPath, QPen, QTransform, QPainter, QColor, QIcon
from PyQt5.QtCore import Qt, QPoint, QPointF

import resources_rc


class ZoomableGraphicsView(QGraphicsView):
    """
    A custom QGraphicsView that enables zooming with the mouse wheel
    and panning with the right mouse button.
    """

    def __init__(self, scene):
        super().__init__(scene)
        self.setRenderHint(QPainter.Antialiasing)
        self.setTransformationAnchor(QGraphicsView.AnchorUnderMouse)
        self.setResizeAnchor(QGraphicsView.AnchorUnderMouse)
        self.panning = False
        self.pan_start_point = QPoint()
        self.setMouseTracking(True)

    def wheelEvent(self, event):
        zoom_factor = 1.2
        if event.angleDelta().y() > 0:
            self.scale(zoom_factor, zoom_factor)
        else:
            self.scale(1 / zoom_factor, 1 / zoom_factor)

    def mousePressEvent(self, event):
        if event.button() == Qt.RightButton:
            self.panning = True
            self.pan_start_point = event.pos()
            self.setCursor(Qt.OpenHandCursor)
            event.accept()
        else:
            super().mousePressEvent(event)

    def mouseMoveEvent(self, event):
        if self.panning:
            delta = event.pos() - self.pan_start_point
            self.pan_start_point = event.pos()
            self.horizontalScrollBar().setValue(self.horizontalScrollBar().value() - delta.x())
            self.verticalScrollBar().setValue(self.verticalScrollBar().value() - delta.y())
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.RightButton:
            self.panning = False
            self.setCursor(Qt.ArrowCursor)
            event.accept()
        else:
            super().mouseReleaseEvent(event)

    def contextMenuEvent(self, event):
        menu = QMenu(self)
        import_action = menu.addAction("Import Image...")

        main_window = self.window()
        if hasattr(main_window, 'import_image'):
            import_action.triggered.connect(main_window.import_image)

        menu.exec_(event.globalPos())


class CellAnalyzerTab(QWidget):
    THRESHOLD_VALUE = 200
    MIN_CONTOUR_AREA = 50
    ERROR_PERCENTAGE = 0.05

    def __init__(self, parent=None, image_width_um=100.0, image_height_um=100.0, status_bar=None):
        super().__init__(parent)
        self.image_width_um = image_width_um
        self.image_height_um = image_height_um
        self.pixel_width_um = 0
        self.pixel_height_um = 0
        self.contours = []

        self.cell_measurements = []
        self.next_cell_id = 1
        self.current_highlight_item = None

        self.draw_line_mode = False
        self.line_start_point = None
        self.temp_line = None
        self.img_color = None
        self.img_gray = None
        self.current_line_items = []

        self.status_bar_message_signal = status_bar.showMessage if status_bar else self.default_status_message

        self.initUI()
        self.update_status("Please import an image or load a project from the 'File' menu.")

    def default_status_message(self, message):
        print(message)

    def initUI(self):
        self.main_layout = QHBoxLayout(self)

        self.scene = QGraphicsScene(self)
        self.view = ZoomableGraphicsView(self.scene)
        self.main_layout.addWidget(self.view)
        self.view.setDragMode(QGraphicsView.NoDrag)
        self.view.viewport().installEventFilter(self)

        self.info_panel = QWidget()
        self.info_layout = QVBoxLayout()
        self.info_panel.setLayout(self.info_layout)

        self.info_panel.setFixedWidth(800)

        title_label = QLabel("<b>Cell Analyzer & Data</b>")
        title_label.setAlignment(Qt.AlignCenter)
        self.info_layout.addWidget(title_label)

        self.mode_group = QWidget(self)
        self.mode_layout = QVBoxLayout(self.mode_group)
        self.analyze_radio = QRadioButton("Area Analysis (Left Click)")
        self.draw_radio = QRadioButton("Line Length (Click & Drag)")
        self.analyze_radio.setChecked(True)
        self.mode_layout.addWidget(self.analyze_radio)
        self.mode_layout.addWidget(self.draw_radio)
        self.info_layout.addWidget(self.mode_group)

        self.width_input_label = QLabel("Image Width (µm):")
        self.width_input = QLineEdit(self)
        self.width_input.setText(str(self.image_width_um))

        self.height_input_label = QLabel("Image Height (µm):")
        self.height_input = QLineEdit(self)
        self.height_input.setText(str(self.image_height_um))

        self.recalculate_button = QPushButton("Recalculate")
        self.recalculate_button.clicked.connect(self.recalculate_dimensions)

        self.auto_scale_button = QPushButton("Auto Scale")
        self.auto_scale_button.clicked.connect(self.auto_scale_image)

        self.info_layout.addWidget(self.width_input_label)
        self.info_layout.addWidget(self.width_input)
        self.info_layout.addWidget(self.height_input_label)
        self.info_layout.addWidget(self.height_input)
        self.info_layout.addWidget(self.recalculate_button)
        self.info_layout.addWidget(self.auto_scale_button)

        self.info_layout.addSpacing(15)

        table_title = QLabel("<b>Measured Cells</b>")
        self.info_layout.addWidget(table_title)

        self.measurements_table = QTableWidget(self)

        self.measurements_table.setMaximumHeight(400)

        self.measurements_table.setColumnCount(4)
        self.measurements_table.setHorizontalHeaderLabels(["ID", "Area (µm²)", "Width (µm)", "Height (µm)"])
        self.measurements_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.measurements_table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.measurements_table.setSelectionMode(QAbstractItemView.SingleSelection)
        self.measurements_table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.measurements_table.cellClicked.connect(self.highlight_cell_from_table)
        self.info_layout.addWidget(self.measurements_table)

        button_layout = QHBoxLayout()

        self.delete_button = QPushButton("Delete Selected")
        self.delete_button.clicked.connect(self.delete_selected_measurement)
        button_layout.addWidget(self.delete_button)

        self.copy_button = QPushButton("Copy to Clipboard")
        self.copy_button.clicked.connect(self.copy_table_to_clipboard)
        button_layout.addWidget(self.copy_button)

        self.info_layout.addLayout(button_layout)

        self.line_length_label = QLabel("Last Line Length: - µm")
        self.line_length_label.setVisible(False)
        self.info_layout.addWidget(self.line_length_label)

        self.info_layout.addStretch(1)

        self.main_layout.addWidget(self.info_panel)

    def import_image(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Import Image",
            "",
            "Image Files (*.png *.jpg *.jpeg *.bmp *.tiff)"
        )
        if file_path:
            self.analyze_image_from_path(file_path)

    def save_state(self):
        if self.img_color is None:
            return None

        img_data = cv2.imencode('.png', cv2.cvtColor(self.img_color, cv2.COLOR_RGB2BGR))[1]
        img_b64 = base64.b64encode(img_data).decode('utf-8')

        tab_widget = self.parent()

        if tab_widget and hasattr(tab_widget, 'tabText'):
            title = tab_widget.tabText(tab_widget.indexOf(self))
        else:
            title = "Untitled Tab"

        serializable_measurements = []
        for m in self.cell_measurements:
            serializable_measurements.append({
                'id': m['id'],
                'area_um2': m['area_um2'],
                'width_um': m['width_um'],
                'height_um': m['height_um'],

                'contour_data': m['contour_data']
            })

        return {
            "title": title,
            "image_b64": img_b64,
            "image_width_um": self.image_width_um,
            "image_height_um": self.image_height_um,
            "cell_measurements": serializable_measurements,
            "next_cell_id": self.next_cell_id,
        }

    def load_state(self, state):
        try:
            img_b64 = state.get("image_b64")
            if not img_b64:
                raise ValueError("No image data found in state.")

            img_data = base64.b64decode(img_b64)
            img_np = np.frombuffer(img_data, np.uint8)
            img = cv2.imdecode(img_np, cv2.IMREAD_COLOR)

            self.image_width_um = state.get("image_width_um", 100.0)
            self.height_input.setText(str(self.image_height_um))
            self.image_height_um = state.get("image_height_um", 100.0)
            self.width_input.setText(str(self.image_width_um))

            self.analyze_image(img)

            self.cell_measurements = []
            self.next_cell_id = state.get("next_cell_id", 1)

            loaded_measurements = state.get("cell_measurements", [])
            for m in loaded_measurements:
                contour_np = np.array(m['contour_data'], dtype=np.int32)

                self.add_measurement_to_table(
                    m['id'],
                    contour_np,
                    m['width_um'],
                    m['height_um'],
                    m['area_um2'],
                    update_id=False
                )

            self.update_status(f"Loaded image into tab with {len(self.cell_measurements)} measurements.")

        except (IOError, ValueError, json.JSONDecodeError) as e:
            QMessageBox.critical(self, "Load Error", f"Failed to load project: {e}")
            self.update_status("Error loading project.")

    def eventFilter(self, source, event):
        if source == self.view.viewport():
            if self.img_color is None:
                return super().eventFilter(source, event)

            if self.analyze_radio.isChecked():
                self.draw_line_mode = False
                self.line_length_label.setVisible(False)
                self.measurements_table.setVisible(True)
                self.delete_button.setVisible(True)
                self.copy_button.setVisible(True)

                if event.type() == event.MouseButtonPress and event.button() == Qt.LeftButton:
                    self.handle_click(event)
            elif self.draw_radio.isChecked():
                self.draw_line_mode = True
                self.line_length_label.setVisible(True)
                self.measurements_table.setVisible(False)
                self.delete_button.setVisible(False)
                self.copy_button.setVisible(False)

                self.clear_highlight()

                if event.type() == event.MouseButtonPress and event.button() == Qt.LeftButton:
                    self.handle_line_press(event)
                elif event.type() == event.MouseMove and event.buttons() == Qt.LeftButton:
                    self.handle_line_move(event)
                elif event.type() == event.MouseButtonRelease and event.button() == Qt.LeftButton:
                    self.handle_line_release(event)
        return super().eventFilter(source, event)

    def handle_click(self, event):
        if event.button() == Qt.LeftButton:
            self.clear_highlight()
            pos = self.view.mapToScene(event.pos())
            x, y = pos.x(), pos.y()

            for contour in self.contours:
                if cv2.pointPolygonTest(contour, (int(x), int(y)), False) >= 0:
                    area_px = cv2.contourArea(contour)
                    area_um2 = area_px * self.pixel_width_um * self.pixel_height_um

                    x_c, y_c, w, h = cv2.boundingRect(contour)
                    width_um = w * self.pixel_width_um
                    height_um = h * self.pixel_height_um

                    self.add_measurement_to_table(
                        self.next_cell_id, contour, width_um, height_um, area_um2
                    )

                    self.update_status(f"Cell {self.next_cell_id} added. Area: {area_um2:.2f} µm²")
                    self.next_cell_id += 1
                    return
        super().mousePressEvent(event)

    def add_measurement_to_table(self, cell_id, contour, width_um, height_um, area_um2, update_id=True):
        """Adds a measurement to the internal list and the QTableWidget."""

        area_error = area_um2 * self.ERROR_PERCENTAGE
        width_error = width_um * self.ERROR_PERCENTAGE
        height_error = height_um * self.ERROR_PERCENTAGE

        outline_item = self.draw_contour(contour, color=Qt.red, line_width=2, opacity=0.8)

        data = {
            'id': cell_id,
            'area_um2': area_um2,
            'width_um': width_um,
            'height_um': height_um,

            'contour_data': contour.tolist(),

            'outline_item': outline_item
        }
        self.cell_measurements.append(data)

        row = self.measurements_table.rowCount()
        self.measurements_table.insertRow(row)

        self.measurements_table.setItem(row, 0, QTableWidgetItem(str(cell_id)))
        self.measurements_table.setItem(row, 1, QTableWidgetItem(f"{area_um2:.2f} ± {area_error:.2f}"))
        self.measurements_table.setItem(row, 2, QTableWidgetItem(f"{width_um:.2f} ± {width_error:.2f}"))
        self.measurements_table.setItem(row, 3, QTableWidgetItem(f"{height_um:.2f} ± {height_error:.2f}"))

        self.measurements_table.scrollToBottom()
        self.measurements_table.selectRow(row)

        self.current_highlight_item = outline_item

    def highlight_cell_from_table(self, row, column):
        """Highlights the cell on the image when its row is clicked in the table."""

        self.clear_highlight()

        item = self.measurements_table.item(row, 0)
        if not item: return

        try:
            cell_id = int(item.text())
        except ValueError:
            return

        data = next((m for m in self.cell_measurements if m['id'] == cell_id), None)

        if data and data['outline_item']:
            item = data['outline_item']

            item.setPen(QPen(QColor(0, 200, 0), 4, Qt.SolidLine))
            item.setOpacity(1.0)
            self.current_highlight_item = item
            self.update_status(f"Highlighted Cell ID: {cell_id}")

    def clear_highlight(self):
        """Resets the style of the previously highlighted item to default red."""
        if self.current_highlight_item and self.current_highlight_item.scene() == self.scene:
            self.current_highlight_item.setPen(QPen(Qt.red, 2, Qt.SolidLine))
            self.current_highlight_item.setOpacity(0.8)
            self.current_highlight_item = None

    def delete_selected_measurement(self):
        """Deletes the selected measurement from the table, list, and scene."""
        selected_rows = self.measurements_table.selectedIndexes()
        if not selected_rows:
            self.update_status("No measurement selected to delete.")
            return

        row = selected_rows[0].row()
        cell_id = int(self.measurements_table.item(row, 0).text())

        data_to_delete = next((m for m in self.cell_measurements if m['id'] == cell_id), None)

        if data_to_delete:

            outline = data_to_delete['outline_item']
            if outline in self.scene.items():
                self.scene.removeItem(outline)

            self.cell_measurements = [m for m in self.cell_measurements if m['id'] != cell_id]

            self.measurements_table.removeRow(row)

            self.update_status(f"Deleted Cell ID: {cell_id}")

            if self.current_highlight_item == outline:
                self.current_highlight_item = None

    def copy_table_to_clipboard(self):
        """Formats the measurement data into a tab-separated string and copies it to the clipboard."""
        if not self.cell_measurements:
            self.update_status("No data to copy.")
            return

        header = ["ID", "Area (um^2) +/- Error", "Width (um) +/- Error", "Height (um) +/- Error"]

        data_string = "\t".join(header) + "\n"

        for data in self.cell_measurements:
            area_w_error = f"{data['area_um2']:.2f} \u00B1 {(data['area_um2'] * self.ERROR_PERCENTAGE):.2f}"
            width_w_error = f"{data['width_um']:.2f} \u00B1 {(data['width_um'] * self.ERROR_PERCENTAGE):.2f}"
            height_w_error = f"{data['height_um']:.2f} \u00B1 {(data['height_um'] * self.ERROR_PERCENTAGE):.2f}"

            row_data = [
                str(data['id']),
                area_w_error,
                width_w_error,
                height_w_error
            ]
            data_string += "\t".join(row_data) + "\n"

        clipboard = QApplication.clipboard()
        clipboard.setText(data_string)
        self.update_status("Measurement data copied to clipboard (tab-separated, ready for Excel).")

    def handle_line_press(self, event):
        self.clear_temp_items()
        self.line_start_point = self.view.mapToScene(event.pos())
        self.temp_line = QGraphicsLineItem()
        self.temp_line.setPen(QPen(Qt.yellow, 2, Qt.SolidLine))
        self.scene.addItem(self.temp_line)

    def handle_line_move(self, event):
        if self.line_start_point:
            end_point = self.view.mapToScene(event.pos())
            self.temp_line.setLine(self.line_start_point.x(), self.line_start_point.y(), end_point.x(),
                                   end_point.y())

    def handle_line_release(self, event):
        if self.line_start_point:
            end_point = self.view.mapToScene(event.pos())

            final_line = QGraphicsLineItem(self.line_start_point.x(), self.line_start_point.y(), end_point.x(),
                                           end_point.y())
            final_line.setPen(QPen(Qt.red, 2, Qt.SolidLine))
            self.scene.addItem(final_line)
            self.current_line_items.append(final_line)

            pixel_length = np.sqrt(
                (end_point.x() - self.line_start_point.x()) ** 2 + (end_point.y() - self.line_start_point.y()) ** 2)

            pixel_size = np.sqrt(self.pixel_width_um * self.pixel_height_um)
            um_length = pixel_length * pixel_size

            self.line_length_label.setText(f"Last Line Length: {um_length:.2f} µm")
            self.update_status(f"Line drawn. Length: {um_length:.2f} µm")

            self.clear_temp_items()

    def recalculate_dimensions(self):
        try:
            self.image_width_um = float(self.width_input.text())
            self.image_height_um = float(self.height_input.text())

            if self.img_gray is not None:
                img_height_px, img_width_px = self.img_gray.shape
                self.pixel_width_um = self.image_width_um / img_width_px
                self.pixel_height_um = self.image_height_um / img_height_px

                self.update_status("Image dimensions updated. Recalculating all measurements...")

                temp_measurements = self.cell_measurements
                self.cell_measurements = []
                self.measurements_table.setRowCount(0)

                for m in temp_measurements:
                    contour_np = np.array(m['contour_data'], dtype=np.int32)

                    x_c, y_c, w_px, h_px = cv2.boundingRect(contour_np)
                    area_px = cv2.contourArea(contour_np)

                    area_um2 = area_px * self.pixel_width_um * self.pixel_height_um
                    width_um = w_px * self.pixel_width_um
                    height_um = h_px * self.pixel_height_um

                    self.add_measurement_to_table(
                        m['id'], contour_np, width_um, height_um, area_um2, update_id=False
                    )

            else:
                self.update_status("No image loaded. Please load an image first.")

        except ValueError:
            self.update_status("Invalid input. Please enter numbers for dimensions.")

    def auto_scale_image(self):
        if self.img_color is not None:
            self.view.fitInView(self.scene.sceneRect(), Qt.KeepAspectRatio)
            self.update_status("Image scaled to fit.")
        else:
            self.update_status("No image to auto-scale.")

    def draw_contour(self, contour, color=Qt.red, line_width=2, opacity=0.8):
        """Draws the contour path on the scene and returns the QGraphicsPathItem."""
        path = QPainterPath()
        path.moveTo(contour[0][0][0], contour[0][0][1])
        for i in range(1, len(contour)):
            path.lineTo(contour[i][0][0], contour[i][0][1])

        if len(contour) > 0:
            path.lineTo(contour[0][0][0], contour[0][0][1])

        path.closeSubpath()

        pen = QPen(color, line_width)
        pen.setCosmetic(True)

        item = QGraphicsPathItem(path)
        item.setPen(pen)
        item.setOpacity(opacity)
        self.scene.addItem(item)
        return item

    def clear_temp_items(self):
        if self.temp_line:
            self.scene.removeItem(self.temp_line)
        self.line_start_point = None
        self.temp_line = None

    def clear_all_outlines(self):
        """Removes all cell outlines and resets measurement state."""

        items_to_remove = [m['outline_item'] for m in self.cell_measurements if 'outline_item' in m]
        for item in items_to_remove:
            if item in self.scene.items():
                self.scene.removeItem(item)

        for item in self.current_line_items:
            if item in self.scene.items():
                self.scene.removeItem(item)
        self.current_line_items = []

        self.cell_measurements = []
        self.measurements_table.setRowCount(0)
        self.next_cell_id = 1
        self.current_highlight_item = None

    def analyze_image(self, img):
        if img is None:
            self.update_status(f"Error: Could not load image from provided data.")
            return

        self.img_color = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(self.img_gray, self.THRESHOLD_VALUE, 255, cv2.THRESH_BINARY_INV)

        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        self.contours = [c for c in contours if cv2.contourArea(c) > self.MIN_CONTOUR_AREA]

        img_height_px, img_width_px, _ = self.img_color.shape
        self.pixel_width_um = self.image_width_um / img_width_px
        self.pixel_height_um = self.image_height_um / img_height_px

        self.scene.clear()
        bytes_per_line = 3 * img_width_px
        qimage = QImage(self.img_color.data, img_width_px, img_height_px, bytes_per_line, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(qimage)
        self.scene.addPixmap(pixmap)

        self.clear_all_outlines()

        self.auto_scale_image()

    def analyze_image_from_path(self, image_path):
        img = cv2.imread(image_path)
        if img is None:
            self.update_status(f"Error: Could not load image from {image_path}")
            return

        self.analyze_image(img)

    def update_status(self, message):
        self.status_bar_message_signal(message)


class CellAnalyzerApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle('Cell Analyzer-v1.0')
        self.setGeometry(100, 100, 1980, 1300)

        self.tabs = QTabWidget()
        self.tabs.setTabsClosable(True)
        self.tabs.tabCloseRequested.connect(self.close_tab)
        self.tabs.tabBarDoubleClicked.connect(self.rename_tab)
        self.setCentralWidget(self.tabs)

        self.add_tab_button = QPushButton("+")
        self.add_tab_button.clicked.connect(self.add_new_tab)
        self.tabs.setCornerWidget(self.add_tab_button)

        self.statusBar = QStatusBar()
        self.setStatusBar(self.statusBar)

        self.create_menus()
        self.add_new_tab()
        self.update_status("Welcome to Cell Analyzer. Use the 'File' menu to get started.")

    def create_menus(self):
        file_menu = self.menuBar().addMenu("File")

        new_tab_action = QAction("New Tab", self)
        new_tab_action.triggered.connect(self.add_new_tab)
        file_menu.addAction(new_tab_action)

        import_action = QAction("Import Image...", self)
        import_action.triggered.connect(self.import_image)
        file_menu.addAction(import_action)

        file_menu.addSeparator()

        save_action = QAction("Save Project...", self)
        save_action.triggered.connect(self.save_project)
        file_menu.addAction(save_action)

        load_action = QAction("Load Project...", self)
        load_action.triggered.connect(self.load_project_with_dialog)
        file_menu.addAction(load_action)

        file_menu.addSeparator()

        exit_action = QAction("Exit", self)
        exit_action.triggered.connect(self.close)
        file_menu.addAction(exit_action)

    def add_new_tab(self, checked=False, title=None):
        try:
            if title is None:
                tab_count = self.tabs.count()
                title = f"New Tab {tab_count + 1}"

            tab_widget = CellAnalyzerTab(status_bar=self.statusBar)
            self.tabs.addTab(tab_widget, title)
            self.tabs.setCurrentWidget(tab_widget)
            self.update_status(f"New tab '{title}' created.")
        except Exception as e:
            QMessageBox.critical(self, "Tab Creation Error", f"Failed to create new tab: {e}")
            self.update_status("Error creating new tab.")

    def close_tab(self, index):
        if self.tabs.count() < 1:
            return

        tab_to_close = self.tabs.widget(index)
        if QMessageBox.question(self, "Close Tab",
                                "Are you sure you want to close this tab? Any unsaved changes will be lost.",
                                QMessageBox.Yes | QMessageBox.No) == QMessageBox.Yes:
            tab_to_close.deleteLater()
            self.tabs.removeTab(index)
            self.update_status(f"Tab closed.")

            if self.tabs.count() == 0:
                self.add_new_tab()

    def rename_tab(self, index):
        old_name = self.tabs.tabText(index)
        new_name, ok = QInputDialog.getText(self, "Rename Tab", "Enter new tab name:", QLineEdit.Normal, old_name)
        if ok and new_name:
            self.tabs.setTabText(index, new_name)
            self.update_status(f"Tab renamed to '{new_name}'.")

    def import_image(self):
        current_tab = self.tabs.currentWidget()
        if isinstance(current_tab, CellAnalyzerTab):
            current_tab.import_image()

    def save_project(self):
        if self.tabs.count() == 0:
            QMessageBox.warning(self, "Save Error", "No tabs are open to save.")
            return

        file_path, _ = QFileDialog.getSaveFileName(
            self,
            "Save Project",
            "",
            "Project Files (*.specdatMV)"
        )
        if not file_path:
            return

        try:
            project_states = []
            for i in range(self.tabs.count()):
                tab_widget = self.tabs.widget(i)
                state = tab_widget.save_state()
                if state:
                    state["title"] = self.tabs.tabText(i)
                    project_states.append(state)

            project_data = {
                "version": "1.0",
                "tabs": project_states
            }

            with open(file_path, 'w') as f:
                json.dump(project_data, f, indent=4)

            self.update_status(f"Project saved to {file_path}")

        except Exception as e:
            QMessageBox.critical(self, "Save Error", f"Failed to save project: {e}")
            self.update_status("Error saving project.")

    def _load_project_from_path(self, file_path):
        """Loads a project directly from a file path without opening a dialog."""
        try:
            with open(file_path, 'r') as f:
                project_data = json.load(f)

            if "tabs" not in project_data:
                raise ValueError("Invalid project file format.")

            while self.tabs.count() > 0:
                self.tabs.removeTab(0)

            for tab_state in project_data["tabs"]:
                new_tab = CellAnalyzerTab(status_bar=self.statusBar)
                new_tab.load_state(tab_state)
                self.tabs.addTab(new_tab, tab_state.get("title", "Loaded Tab"))

            self.update_status(f"Project loaded from {file_path}")
            return True

        except (IOError, ValueError, json.JSONDecodeError) as e:
            QMessageBox.critical(self, "Load Error", f"Failed to load project: {e}")
            self.update_status("Error loading project.")
            return False

    def load_project_with_dialog(self):
        """Opens a file dialog and calls the private load method."""
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Load Project",
            "",
            "Project Files (*.specdatMV);;JSON Files (*.json);;All Files (*)"
        )
        if not file_path:
            return

        self._load_project_from_path(file_path)

    def update_status(self, message):
        self.statusBar.showMessage(message)


if __name__ == '__main__':

    if sys.platform == 'win32':
        try:
            ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID('dataviewer2D.CellAnalyzerApp.1.0')
        except AttributeError:
            pass

    app = QApplication(sys.argv)

    app.setStyleSheet("QWidget { font-size: 9pt; }")

    app.setWindowIcon(QIcon(':/icons/icon.ico'))

    main_window = CellAnalyzerApp()
    main_window.show()

    if len(sys.argv) > 1:
        file_to_open = sys.argv[1]

        if os.path.exists(file_to_open) and file_to_open.lower().endswith(('.specdatmv', '.json')):
            try:

                main_window._load_project_from_path(file_to_open)
            except Exception as e:

                QMessageBox.critical(main_window, "Error Opening Project",
                                     f"Failed to load project from '{file_to_open}': {e}")
        else:
            QMessageBox.warning(main_window, "Unsupported File",
                                f"The file '{file_to_open}' is not a recognized project file (.specdatMV or .json).")

    sys.exit(app.exec_())
