import sys
import psutil
from PyQt5.QtWidgets import QApplication, QWidget, QVBoxLayout, QLabel, QPushButton, QToolBar, QAction, QGroupBox, QHBoxLayout
from PyQt5.QtCore import QTimer

class MainWindow(QWidget):
    def __init__(self):
        super().__init__()

        # Retrieve the screen geometry
        screen_geometry = app.desktop().screenGeometry()
        screen_width = screen_geometry.width()
        screen_height = screen_geometry.height()
        
        # Set window properties
        self.setWindowTitle('GUI')
        window_width = int(screen_width * 0.8)  # Limit width to 80% of screen width
        window_height = int(screen_height * 0.8)  # Limit height to 80% of screen height
        self.setGeometry((screen_width - window_width) // 2, (screen_height - window_height) // 2, window_width, window_height)  # Center the window on the screen

        # Create a QVBoxLayout to arrange widgets vertically
        layout = QVBoxLayout()

        # Create a toolbar
        toolbar = QToolBar()
        toolbar.setFixedHeight(40)  # Set the fixed height of the toolbar
        layout.addWidget(toolbar)

        # Add actions to the toolbar
        action1 = QAction("Action 1", self)
        toolbar.addAction(action1)
        action2 = QAction("Action 2", self)
        toolbar.addAction(action2)

        # Create a QPushButton to quit the application
        quit_button = QPushButton("Quitter")
        quit_button.clicked.connect(app.quit)
        layout.addWidget(quit_button)

        # Create a group box to display system information
        group_box = QGroupBox("System Information")
        layout.addWidget(group_box)

        # Create a QHBoxLayout for the group box
        group_layout = QVBoxLayout(group_box)

        # Create a QLabel to display CPU percentage
        self.cpu_label = QLabel()
        group_layout.addWidget(self.cpu_label)

        # Create a QLabel to display mem usage
        self.mem_label = QLabel()
        group_layout.addWidget(self.mem_label)

        # Set the layout for the group box
        group_box.setLayout(group_layout)

        # Set the layout for the window
        self.setLayout(layout)

        # Create a QTimer to update CPU percentage label every second
        self.timer_cpu = QTimer(self)
        self.timer_cpu.timeout.connect(self.update_cpu_label)
        self.timer_cpu.start(1000)  # Update every 1000 milliseconds (1 second)

        # Create a QTimer to update mem usage label every second
        self.timer_mem = QTimer(self)
        self.timer_mem.timeout.connect(self.update_mem_label)
        self.timer_mem.start(1000)  # Update every 1000 milliseconds (1 second)

        # Initial update of the CPU label
        self.update_cpu_label()

        # Initial update of the CPU label
        self.update_mem_label()

    def update_cpu_label(self):
        # Get CPU percentage
        cpu_percent = psutil.cpu_percent(interval=None)
        self.cpu_label.setText(f"CPU Usage: {cpu_percent}%")

    def update_mem_label(self):
        # Get CPU percentage
        memory_info = psutil.virtual_memory()  # Get virtual memory usage
        used_memory = round(memory_info.used / (1024 ** 3), 2)
        self.mem_label.setText(f"Memory Usage: {used_memory} GB")

def main():

    global app
    # Create an instance of QApplication
    app = QApplication([])
    
    # Create a QWidget object, which serves as the main window
    window = MainWindow()
    
    # Display the window
    window.show()

    # Execute the application's main event loop
    sys.exit(app.exec_())

if __name__ == '__main__':
    main()
