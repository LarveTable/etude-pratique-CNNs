import sys
from PyQt5.QtWidgets import QApplication, QWidget, QVBoxLayout, QDesktopWidget

def main():
    # Create an instance of QApplication
    app = QApplication([])
    
    # Create a QWidget object, which serves as the main window
    window = QWidget()

    # Retrieve the screen geometry
    screen_geometry = app.desktop().screenGeometry()
    screen_width = screen_geometry.width()
    screen_height = screen_geometry.height()
    
    # Set window properties
    window.setWindowTitle('GUI')
    window_width = int(screen_width * 0.8)  # Limit width to 80% of screen width
    window_height = int(screen_height * 0.8)  # Limit height to 80% of screen height
    window.setGeometry((screen_width - window_width) // 2, (screen_height - window_height) // 2, window_width, window_height)  # Center the window on the screen
    
    # Display the window
    window.show()
    
    # Execute the application's main event loop
    sys.exit(app.exec_())

if __name__ == '__main__':
    main()
