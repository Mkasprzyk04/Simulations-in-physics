from projektcore import run_simulation
import sys
from scipy.sparse import diags
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget,
    QVBoxLayout, QHBoxLayout, QLabel,
    QLineEdit, QPushButton, QMessageBox, 
    QTextBrowser, QStackedWidget
)
import numpy as np
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.figure import Figure
from PySide6.QtCore import QTimer, Qt
from PySide6.QtGui import QPainter, QPixmap
import os 

button_fontsize = "28px;"
button_padding = "10px;"
button_marginbottom = "6px;" 
button_width = 400


# Define the class corresponding to the start screen
class StartScreen(QWidget):
    def __init__(self, stack):
        super().__init__()
        self.stack = stack
        # Part responsible for importing the background image of the main menu
        img_path = os.path.join(os.path.dirname(__file__), "graphics", "Kwanty_ekran_startowy.jpg")
        img_path = img_path.replace("\\", "/")
        self.bg_pixmap = QPixmap(img_path)
        layout = QVBoxLayout()
        layout.setAlignment(Qt.AlignmentFlag.AlignCenter)

        # Centered title
        title = QLabel("Quantator 5000")
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        title.setStyleSheet("font-size: 230px; font-weight: bold; margin-bottom: 30px;")
        layout.addWidget(title)

        # Subtitle, also centered
        label = QLabel("Choose an option:")
        label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        label.setStyleSheet("font-size: 100px; margin-bottom: 20px;")
        layout.addWidget(label)

        # Define utility buttons and margins, styling
        btn_layout = QVBoxLayout()
        btn_layout.setAlignment(Qt.AlignmentFlag.AlignHCenter)
        btn_sim = QPushButton("Quantum simulation")
        btn_ins = QPushButton("User manual")
        btn_exit = QPushButton("Exit")
        
        buttons = [btn_sim, btn_ins, btn_exit]
        for btn in buttons:
            btn.setFixedWidth(button_width)
            btn.setStyleSheet("font-size: " + button_fontsize + "padding: " + button_padding + "margin-bottom: " + button_marginbottom)
            btn_layout.addWidget(btn)
        layout.addLayout(btn_layout)

        # Define button functionality
        self.setLayout(layout)
        btn_sim.clicked.connect(lambda: self.stack.setCurrentIndex(1))
        btn_ins.clicked.connect(lambda: self.stack.setCurrentIndex(2))
        btn_exit.clicked.connect(QApplication.instance().quit)


    # Method responsible for drawing the background
    def paintEvent(self, event):
        painter = QPainter(self)
        if not self.bg_pixmap.isNull():
            painter.drawPixmap(self.rect(), self.bg_pixmap)
        super().paintEvent(event)

# Define the class corresponding to the simulation window
class Simulation(QMainWindow):
    def __init__(self,stack):
        super().__init__()
        self.stack = stack

        # Define the layout
        container = QWidget()
        layout = QVBoxLayout()
        container.setLayout(layout)
        self.setCentralWidget(container)

        # Aesthetics of the control panel, options for entering numbers into it, and assigning functionality
        ctrl = QHBoxLayout()
        self.edit_gamma = QLineEdit("0.0"); ctrl.addWidget(QLabel("γ:")); ctrl.addWidget(self.edit_gamma)
        self.edit_omega = QLineEdit("0.0"); ctrl.addWidget(QLabel("ω:")); ctrl.addWidget(self.edit_omega)
        self.edit_periods = QLineEdit("0.0"); ctrl.addWidget(QLabel("Number of periods:")); ctrl.addWidget(self.edit_periods)
        self.btn_run = QPushButton("Run"); self.btn_run.clicked.connect(self.on_run); ctrl.addWidget(self.btn_run)
        layout.addLayout(ctrl)

        # Space for the plot
        self.fig = Figure(figsize=(5,3))
        self.canvas = FigureCanvasQTAgg(self.fig)
        layout.addWidget(self.canvas)
        
        # Adding buttons used in this panel and appending them to the layout
        btn_layout = QVBoxLayout()
        btn_layout.setAlignment(Qt.AlignmentFlag.AlignHCenter)
        btn_back = QPushButton("Return to main menu")
        self.btn_reset = QPushButton("Reset animation")

        buttons =[btn_back, self.btn_reset]
        for btn in buttons:
            btn.setFixedWidth(button_width)
            btn.setStyleSheet("font-size: "  + button_fontsize +  "padding: " + button_padding)
            btn_layout.addWidget(btn)
        layout.addLayout(btn_layout)

        # Define button functionality and hide the animation reset button
        self.btn_reset.setVisible(False) 
        self.setLayout(layout)
        btn_back.clicked.connect(lambda: self.stack.setCurrentIndex(0))
        self.btn_reset.clicked.connect(self.on_screen_reset)

        # Functions responsible for animation
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_frame)
        self.frame_idx = 0
    
    # Functions used in widgets
    def on_run(self):
        # Disable the run button so the user doesn't mess up
        self.btn_run.setEnabled(False)
        # Convert the entered text to float and throw an error if it's not possible
        try:
            gamma = float(self.edit_gamma.text())
            omega = float(self.edit_omega.text())
            periods = float(self.edit_periods.text())
            if periods <= 0:
                QMessageBox.critical(self, "Error", "Please reconsider, the number of periods is invalid!")
                return
            if omega == 0 or gamma == 0:
                QMessageBox.critical(self, "Error", "No time dependence")
                return
        except ValueError:
            QMessageBox.critical(self, "Error", "Invalid parameter values!")
            return

        # Declaration of variables used in the animation
        self.x, self.psi2, self.V = run_simulation(gamma, omega, periods)
        self.ax = self.fig.subplots()
        self.ax.clear()
        self.ax2 = self.ax.twinx()
        self.ax2.clear()

        # Draw psi on the left axis at the initial time
        self.line_psi, = self.ax.plot(self.x, self.psi2[0], label='|ψ(x,t)|²', color='b')
        self.ax.set_ylabel("", color='b', size=20)
        self.ax.tick_params(axis='y', labelcolor='b')
        psi_max = float(np.max(self.psi2))
        self.ax.set_ylim(0, 1.05*psi_max)

        # Draw V on the right axis at the initial time
        self.line_V, = self.ax2.plot(self.x, self.V[0], label='V(x,t)', color='r')
        self.ax2.set_ylabel("", color='r', size=20)
        self.ax2.tick_params(axis='y', labelcolor='r')

        # Define a single legend outside the plot
        lines = [self.line_psi, self.line_V]
        labels = [line.get_label() for line in lines]
        self.ax.legend(lines, labels, loc='center left', bbox_to_anchor=(1.02, 0.5),fontsize = "large")

        # X-axis description
        self.ax.set_xlabel("x",fontsize = 20)
        self.canvas.draw()
        self.timer.start(100)

    # Function responsible for the animation
    def update_frame(self):

        # Element responsible for stopping the animation
        if self.frame_idx >= len(self.psi2):
            self.timer.stop()
            self.btn_reset.setVisible(True)   # Shows the reset button after the animation finishes
            return
        
        # Code responsible for updating the image and jumping one frame in the animation
        self.line_psi.set_ydata(self.psi2[self.frame_idx])
        self.line_V.set_ydata(self.V[self.frame_idx])
        self.ax.set_title(f"Simulation progress: {self.frame_idx + 1}%")
        self.canvas.draw()
        self.frame_idx += 1
    
    def on_screen_reset(self):
        # Stop the timer, which shouldn't happen, but we do it just in case
        if self.timer.isActive():
            self.timer.stop()

        # Clear all indices and remove plots
        self.frame_idx = 0
        if hasattr(self, 'psi2'):
            self.ax.remove()
            self.ax2.remove()
        # Get space for the plot, hide the reset button, remove the button, and allow the animation to restart
        self.canvas.draw()
        self.btn_reset.hide()
        self.btn_run.setEnabled(True)  

# Define the class corresponding to the instruction window
class Instruction(QMainWindow):
    def __init__(self, stack):
        super().__init__()
        self.stack = stack
        # Graphical part responsible for the background styling
        img_path = os.path.join(os.path.dirname(__file__), "graphics", "Kwanty_instruktażowy.jpeg")
        img_path = img_path.replace("\\", "/")
        self.bg_pixmap = QPixmap(img_path)

        # Define the layout and container
        container = QWidget()
        layout = QVBoxLayout()
        container.setLayout(layout)
        self.setCentralWidget(container)

        # Centered title
        title = QLabel("User manual :)")
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        title.setStyleSheet("font-size: 100px; font-weight: bold; color: red;")
        layout.addWidget(title)

        # Prepare space for the instructions and set its appearance
        self.text_browser = QTextBrowser()
        self.text_browser.setStyleSheet("background: transparent; border: none;")
        self.text_browser.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground, True)
        self.text_browser.viewport().setAutoFillBackground(False)
        
        # Formatted instruction text in HTML; this language was used because it is automatically rendered by QTextBrowser
        self.text_browser.setHtml("""
        <div style="background-color: black; color: white; text-align: center; font-size: 40px; padding: 20px;">
            <ul style="list-style-position: inside; padding-left: 4;">
                <li>This program shows the evolution of a wave function in the form of a Gaussian wave packet with zero momentum and a coefficient of σ = 1 in a potential of the form V(x,t) = -0.5x<sup>2</sup> + γcos(ωt)x<sup>3</sup> + 0.0625x<sup>4</sup></li>
                <li>Click the "quantum simulation" button located in the main menu.</li>
                <li>This button will take you to a panel where you can select the parameters describing the time evolution and the number of periods you want to observe.</li>
                <li>Enter the data (please be gentle when choosing your parameters).</li>
                <li>Next, click the "run" button.</li>
                <li>After the animation finishes, if you want to replay it, click the "reset animation" button. You can change the parameters before repeating the animation.</li>        
            </ul>
            <p><span style="color: white;">Warmest regards!</span></p>
        </div>
        """)
        self.text_browser.setFixedHeight(500)
        layout.addWidget(self.text_browser)

        # Return to main menu button
        btn_back = QPushButton("Return to main menu")
        btn_back.setFixedWidth(button_width)
        btn_back.setStyleSheet("font-size: "  + button_fontsize +  "padding: " + button_padding)
        btn_layout = QVBoxLayout()
        btn_layout.setAlignment(Qt.AlignmentFlag.AlignBottom)
        btn_layout.addWidget(btn_back)
        layout.addLayout(btn_layout)
        btn_back.clicked.connect(lambda: self.stack.setCurrentIndex(0))
        
    # Paint the instruction background
    def paintEvent(self, event):
        painter = QPainter(self)
        if not self.bg_pixmap.isNull():
            painter.drawPixmap(self.rect(), self.bg_pixmap)
        super().paintEvent(event)

# Main application manager that connects all previously defined classes
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.stack = QStackedWidget()
        self.stack.addWidget(StartScreen(self.stack))  # index 0
        self.stack.addWidget(Simulation(self.stack))  # index 1
        self.stack.addWidget(Instruction(self.stack))  # index 2
        self.setCentralWidget(self.stack)
        self.setWindowTitle("Kwantator 5000")

# Show the main window
if __name__ == "__main__":
    app = QApplication(sys.argv)
    mw = MainWindow()
    mw.showMaximized()    
    sys.exit(app.exec())