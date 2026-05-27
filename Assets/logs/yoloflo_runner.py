from yoloflo_core import run_yoloflo

class YoloFLO:
    def __init__(self):
        print("YoloFLO initialized.")

    def run(self):
        print("Running YoloFLO pipeline...")
        res = run_yoloflo()
        print("Done.")
        return res


# ---- one-line execution ----
if __name__ == "__main__":
    model = YoloFLO()
    res = model.run()
    print(res)
