import sys
import numpy as np
import sympy as sp
from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget, QTabWidget, QVBoxLayout, QHBoxLayout,
                             QLabel, QLineEdit, QPushButton, QComboBox, QTextEdit, QFileDialog,
                             QMessageBox, QRadioButton, QButtonGroup)
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

x_symb = sp.symbols('x')

equations = [
    {
        'name': 'x^2 - 2',
        'f': lambda x: x ** 2 - 2,
        'symb_expr': x_symb ** 2 - 2
    },
    {
        'name': 'sin(x)',
        'f': lambda x: np.sin(x),
        'symb_expr': sp.sin(x_symb)
    },
    {
        'name': 'x^3 + 2.28x^2 - 1.934x - 3.907',
        'f': lambda x: x ** 3 + 2.28 * x ** 2 - 1.934 * x - 3.907,
        'symb_expr': x_symb ** 3 + 2.28 * x_symb ** 2 - 1.934 * x_symb - 3.907
    }
]

systems = [
    {
        'name': 'x^2 + y^2 = 1; y = x^3',
        'f1': lambda x, y: x ** 2 + y ** 2 - 1,
        'f2': lambda x, y: y - x ** 3,
        'jacobian': lambda x, y: np.array([
            [2 * x, 2 * y],
            [-3 * x ** 2, 1]
        ])
    },
    {
        'name': 'x + y^2 = 4; x^2 - y = 1',
        'f1': lambda x, y: x + y ** 2 - 4,
        'f2': lambda x, y: x ** 2 - y - 1,
        'jacobian': lambda x, y: np.array([
            [1, 2 * y],
            [2 * x, -1]
        ])
    }
]

def derivative(f_s, symb):
    return sp.lambdify(symb, sp.diff(f_s, symb))

def check_for_singular_root(f, a, b):
    step = (b - a) / 100
    cnt = 0
    last = f(a)
    i = a + step
    while i <= b:
        if last == 0:
            cnt += 1
        current = f(i)
        if current * last < 0:
            cnt += 1
        last = current
        i += step
    if f(b) == 0:
        cnt += 1
    return cnt == 1


def chord_method(f, a, b, epsilon, max_iter=10000):
    fa, fb = f(a), f(b)
    iterations = 0
    while abs(b - a) > epsilon and iterations < max_iter:
        iterations += 1
        x = (a * fb - b * fa) / (fb - fa)
        fx = f(x)
        if abs(fx) <= epsilon:
            return x, fx, iterations
        if fa * fx < 0:
            b, fb = x, fx
        else:
            a, fa = x, fx

    return x, f(x), iterations


def secant_method(f, a, b, epsilon, max_iter=10000):
    x0 = (a + b) / 2
    x1 = x0 + epsilon
    iterations = 0
    f0, f1 = f(x0), f(x1)
    while abs(f1) > epsilon and iterations < max_iter:
        x2 = x1 - f1 * (x1 - x0) / (f1 - f0)
        f0, f1 = f(x1), f(x2)
        x0, x1 = x1, x2
        iterations += 1
    return x1, f1, iterations


def simple_iteration(f, phi, x0, epsilon, max_iter=10000):
    iterations = 0
    while iterations < max_iter:
        iterations += 1
        x1 = phi(x0)
        if abs(x1 - x0) <= epsilon:
            break
        x0 = x1
    return x1, f(x1), iterations


def newton_system(f1, f2, jacobian, x0, y0, epsilon, max_iter=10000):
    x = np.array([x0, y0], dtype=float)
    errors = []
    iterations = 0
    while iterations < max_iter:
        iterations += 1
        J = jacobian(x[0], x[1])
        F = np.array([f1(x[0], x[1]), f2(x[0], x[1])])
        try:
            delta = np.linalg.solve(J, -F)
        except np.linalg.LinAlgError:
            break
        x_new = x + delta
        error = max(abs(delta[0]), abs(delta[1]))
        errors.append(error)
        if error < epsilon:
            break
        x = x_new
    return x[0], x[1], iterations, errors


class EquationTab(QWidget):
    def __init__(self):
        super().__init__()
        self.initUI()

    def initUI(self):
        layout = QVBoxLayout()

        self.equation_combo = QComboBox()
        for eq in equations:
            self.equation_combo.addItem(eq['name'])
        layout.addWidget(QLabel('Выберите уравнение:'))
        layout.addWidget(self.equation_combo)

        # Ввод интервала и точности
        self.a_input = QLineEdit()
        self.b_input = QLineEdit()
        self.epsilon_input = QLineEdit()

        layout.addWidget(QLabel('Интервал [a, b]:'))
        h_layout = QHBoxLayout()
        h_layout.addWidget(self.a_input)
        h_layout.addWidget(self.b_input)
        layout.addLayout(h_layout)

        layout.addWidget(QLabel('Погрешность:'))
        layout.addWidget(self.epsilon_input)

        # Загрузка из файла
        self.load_button = QPushButton('Загрузить данные из файла')
        self.load_button.clicked.connect(self.load_from_file)
        layout.addWidget(self.load_button)

        # Выбор метода
        self.method_group = QButtonGroup()
        self.chord_radio = QRadioButton('Метод хорд')
        self.secant_radio = QRadioButton('Метод секущих')
        self.iteration_radio = QRadioButton('Метод простой итерации')
        self.method_group.addButton(self.chord_radio)
        self.method_group.addButton(self.secant_radio)
        self.method_group.addButton(self.iteration_radio)

        layout.addWidget(QLabel('Выберите метод:'))
        layout.addWidget(self.chord_radio)
        layout.addWidget(self.secant_radio)
        layout.addWidget(self.iteration_radio)

        # Кнопка расчета
        self.calculate_button = QPushButton('Решить уравнение')
        self.calculate_button.clicked.connect(self.calculate)
        layout.addWidget(self.calculate_button)

        # Вывод результатов
        self.result_output = QTextEdit()
        self.result_output.setReadOnly(True)
        layout.addWidget(self.result_output)

        # Сохранение результатов
        self.save_button = QPushButton('Сохранить результаты в файл')
        self.save_button.clicked.connect(self.save_results)
        layout.addWidget(self.save_button)

        # График
        self.figure = Figure()
        self.canvas = FigureCanvas(self.figure)
        layout.addWidget(self.canvas)

        self.setLayout(layout)

    def load_from_file(self):
        filename, _ = QFileDialog.getOpenFileName(self, 'Выберите файл', '', 'Text Files (*.txt)')
        if filename:
            with open(filename, 'r') as f:
                data = f.read().split()
                try:
                    self.a_input.setText(data[0])
                    self.b_input.setText(data[1])
                    self.epsilon_input.setText(data[2])
                except:
                    QMessageBox.warning(self, 'Error', 'Ошибка при чтении файла.')

    def save_results(self):
        filename, _ = QFileDialog.getSaveFileName(self, 'Выберите файл для сохранения', '', 'Text Files (*.txt)')
        if filename:
            with open(filename, 'w') as f:
                f.write(self.result_output.toPlainText())

    def calculate(self):
        try:
            a = float(self.a_input.text())
            b = float(self.b_input.text())
            epsilon = float(self.epsilon_input.text())
            if a >= b:
                raise ValueError('a должно быть < b!')
            if epsilon <= 0:
                raise ValueError('Погрешность должна быть > 0!')
            if b - a <= epsilon:
                raise ValueError('Погрешность должна быть меньше, чем длина интервала!')
        except ValueError as e:
            QMessageBox.warning(self, 'Input Error', str(e))
            return

        eq_index = self.equation_combo.currentIndex()
        equation = equations[eq_index]
        f = equation['f']

        if not check_for_singular_root(f, a, b):
            QMessageBox.warning(self, 'Error', f'Нет корней или их несколько на заданном интервале [{a}, {b}]')
            return


        if self.chord_radio.isChecked():
            method = 'chord'
        elif self.secant_radio.isChecked():
            method = 'secant'
        elif self.iteration_radio.isChecked():
            method = 'iteration'
        else:
            QMessageBox.warning(self, 'Error', 'Метод не выбран!')
            return

        if method == 'iteration':
            f_s = equation['symb_expr']
            d_f = derivative(f_s, x_symb)
            lmbd = 1 / max(abs(d_f(a)), abs(d_f(b)))
            if d_f((a + b) / 2) > 0:
                lmbd *= -1
            phi = lambda x: x + lmbd * f(x)
            phi_s = x_symb + lmbd * f_s
            d_phi = derivative(phi_s, x_symb)
            max_d_phi = max(abs(d_phi(a)), abs(d_phi(b)))
            if max_d_phi >= 1:
                QMessageBox.warning(self, 'Error', 'Условие сходимости не выполняется!')
                return
            x0 = (a + b) / 2
            root, func_val, iterations = simple_iteration(f, phi, x0, epsilon)
        elif method == 'chord':
            root, func_val, iterations = chord_method(f, a, b, epsilon)
        elif method == 'secant':
            root, func_val, iterations = secant_method(f, a, b, epsilon)

        if root is None:
            QMessageBox.warning(self, 'Error', 'Метод не смог сойтись, нет решения.')
            return

        result_text = f'Найденный корень: {root:.6f}\n'
        result_text += f'Значение функции в корне: {func_val:.6f}\n'
        result_text += f'Кол-во итераций: {iterations}'
        self.result_output.setText(result_text)

        # Обновление графика
        self.figure.clear()
        ax = self.figure.add_subplot(111)
        x_plot = np.linspace(a, b, 400)
        y_plot = f(x_plot)
        ax.plot(x_plot, y_plot)
        ax.axhline(0, color='black', linewidth=0.5)
        ax.axvline(root, color='red', linestyle='--', label='Root')
        ax.scatter(root, 0, color='red')
        ax.set_xlabel('x')
        ax.set_ylabel('f(x)')
        ax.legend()
        ax.grid()
        self.canvas.draw()


class SystemTab(QWidget):
    def __init__(self):
        super().__init__()
        self.initUI()

    def initUI(self):
        layout = QVBoxLayout()

        # Выбор системы
        self.system_combo = QComboBox()
        for sys in systems:
            self.system_combo.addItem(sys['name'])
        layout.addWidget(QLabel('Выберите систему уравнений:'))
        layout.addWidget(self.system_combo)

        # Ввод начальных приближений
        self.x0_input = QLineEdit()
        self.y0_input = QLineEdit()
        self.epsilon_input = QLineEdit()

        layout.addWidget(QLabel('Начальное приближение (x0, y0):'))
        h_layout = QHBoxLayout()
        h_layout.addWidget(self.x0_input)
        h_layout.addWidget(self.y0_input)
        layout.addLayout(h_layout)

        layout.addWidget(QLabel('Погрешность:'))
        layout.addWidget(self.epsilon_input)

        # Кнопка расчета
        self.calculate_button = QPushButton('Решить систему уравнений')
        self.calculate_button.clicked.connect(self.calculate)
        layout.addWidget(self.calculate_button)

        # Вывод результатов
        self.result_output = QTextEdit()
        self.result_output.setReadOnly(True)
        layout.addWidget(self.result_output)

        # График
        self.figure = Figure()
        self.canvas = FigureCanvas(self.figure)
        layout.addWidget(self.canvas)

        self.setLayout(layout)

    def calculate(self):
        try:
            x0 = float(self.x0_input.text())
            y0 = float(self.y0_input.text())
            epsilon = float(self.epsilon_input.text())
            if epsilon <= 0:
                raise ValueError('Погрешность должна быть положительной!')
        except ValueError as e:
            QMessageBox.warning(self, 'Input Error', str(e))
            return

        sys_index = self.system_combo.currentIndex()
        system = systems[sys_index]
        f1 = system['f1']
        f2 = system['f2']
        jacobian = system['jacobian']

        x, y, iterations, errors = newton_system(f1, f2, jacobian, x0, y0, epsilon)

        result_text = f'Solution: x = {x:.6f}, y = {y:.6f}\n'
        result_text += f'Кол-во итераций: {iterations}\n'
        result_text += 'Вектор погрешностей:\n' + '[' + '; '.join([f'{e:.6f}' for e in errors]) + ']\n'
        self.result_output.setText(result_text)

        # Обновление графика
        self.figure.clear()
        ax = self.figure.add_subplot(111)

        x_vals = np.linspace(x - 3, x + 3, 100)
        y_vals = np.linspace(y - 3, y + 3, 100)
        X, Y = np.meshgrid(x_vals, y_vals)
        Z1 = f1(X, Y)
        Z2 = f2(X, Y)

        ax.contour(X, Y, Z1, levels=[0], colors='r')
        ax.contour(X, Y, Z2, levels=[0], colors='b')
        ax.scatter(x, y, color='green', label='Solution')
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.legend()
        ax.grid()
        self.canvas.draw()


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle('Лабораторная №2')
        self.tabs = QTabWidget()
        self.setCentralWidget(self.tabs)

        self.equation_tab = EquationTab()
        self.system_tab = SystemTab()

        self.tabs.addTab(self.equation_tab, 'Нелинейные уравнения')
        self.tabs.addTab(self.system_tab, 'Системы нелинейных ур-й')


if __name__ == '__main__':
    app = QApplication(sys.argv)
    main = MainWindow()
    main.show()
    sys.exit(app.exec())