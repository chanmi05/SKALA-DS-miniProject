"""공통 유틸 : 경로, 로그 기록, 그림 저장, 배치 색상."""
import os
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(ROOT, 'data')
PROCESSED_DIR = os.path.join(DATA_DIR, 'processed')
RESULTS_DIR = os.path.join(ROOT, 'results')

BATCHES = ['B1', 'B2', 'B3']
BCOLOR = {'B1': '#1f77b4', 'B2': '#ff7f0e', 'B3': '#2ca02c'}

plt.rcParams.update({'figure.dpi': 110, 'font.size': 10,
                     'axes.spines.top': False, 'axes.spines.right': False})


class Reporter:
    """화면 출력 + 텍스트 로그 + 그림 저장을 한 곳에서 관리.

    notebook마다 Reporter('01_EDA') 처럼 만들면
    results/01_EDA/figs/*.png, results/01_EDA/log.txt 에 결과가 쌓인다.
    """

    def __init__(self, name):
        self.dir = os.path.join(RESULTS_DIR, name)
        self.fig_dir = os.path.join(self.dir, 'figs')
        os.makedirs(self.fig_dir, exist_ok=True)
        self.log_path = os.path.join(self.dir, 'log.txt')
        open(self.log_path, 'w').close()

    def log(self, *args):
        text = ' '.join(str(a) for a in args)
        print(text)
        with open(self.log_path, 'a') as fp:
            fp.write(text + '\n')

    def table(self, df, title=None, fmt='{:.4g}'):
        if title:
            self.log(f'\n[{title}]')
        self.log(df.to_string(float_format=lambda x: fmt.format(x)))

    def section(self, title):
        self.log('\n' + '=' * 78 + f'\n{title}\n' + '=' * 78)

    def savefig(self, name):
        plt.savefig(os.path.join(self.fig_dir, f'{name}.png'), dpi=160, bbox_inches='tight')
        plt.show()

    def zip_and_download(self):
        import shutil
        path = shutil.make_archive(self.dir, 'zip', self.dir)
        print('저장 :', path)
        try:
            from google.colab import files
            files.download(path)
        except Exception:
            pass
        return path


def batch_legend(ax, **kw):
    handles = [Line2D([0], [0], color=BCOLOR[b], lw=2, label=b) for b in BATCHES]
    ax.legend(handles=handles, **kw)
