![hypesquad](https://github.com/boostcampaitech5/level2_cv_datacentric-cv-15/assets/113939970/729c72fc-7159-4f92-b42c-5b9beaee8b7d)
## 📸 병원 진료비 영수증 OCR Task
![ocr](https://github.com/boostcampaitech5/level2_cv_datacentric-cv-15/assets/113939970/c35033e6-46e2-4dd7-9cc6-728405984f8c)
스마트폰으로 카드를 결제하거나, 카메라로 카드를 인식할 경우 자동으로 카드 번호가 입력되는 경우가 있습니다. 또 주차장에 들어가면 차량 번호가 자동으로 인식되는 경우도 흔히 있습니다. 이처럼 OCR (Optimal Character Recognition) 기술은 사람이 직접 쓰거나 이미지 속에 있는 문자를 얻은 다음 이를 컴퓨터가 인식할 수 있도록 하는 기술로, 컴퓨터 비전 분야에서 현재 널리 쓰이는 대표적인 기술 중 하나입니다.

### **📆** 대회 일정 : 2023.05.24 ~ 2023.06.01

### **🗂️** Dataset

---

- 전체 이미지 개수 : 200장 (train 100장 + test 100장)
- annotation : UFO format (Upstage Format for OCR)
- input : 병원 진료비 영수증 이미지, annotation file (UFO format)
- output : 글자 위치 검출 박스

## 👨🏻‍💻 👩🏻‍💻 팀 구성

-------------
|![logo1](https://github.com/boostcampaitech5/level2_cv_datacentric-cv-15/assets/113939970/25c7087d-ed59-474b-b1fe-2adba3689faa)|![logo2](https://github.com/boostcampaitech5/level2_cv_datacentric-cv-15/assets/113939970/1086d3c7-a79c-4f43-a883-bdc31b68863d)|![logo3](https://github.com/boostcampaitech5/level2_cv_datacentric-cv-15/assets/113939970/1699dc25-c208-4509-b0a5-50d4f3b04326)|![logo4](https://github.com/boostcampaitech5/level2_cv_datacentric-cv-15/assets/113939970/0c5306da-f5ea-4b8c-996a-092d74ae0836)|![logo5](https://github.com/boostcampaitech5/level2_cv_datacentric-cv-15/assets/113939970/9dd39e5f-ebf7-4a6f-a478-e87e6dcef317)|
| :---: | :---: | :---: | :---: |  :---: |
| [김용우](https://github.com/yongwookim1) | [박종서](https://github.com/justinpark820) | [서영덕](https://github.com/SeoYoungDeok) |[신현준](https://github.com/june95) |[조수혜](https://github.com/suhyehye) |

## 📊 EDA 결과

---

- train dataset에 annotation이 잘못 되어있는 박스가 존재
- 화강석 배경 또는 배경의 그림자에서 박스가 잘못 검출되는 경우가 존재
- QR code에서 박스가 잘못 검출되는 경우가 존재
- 수직으로 쓰여있는 글자를 잘 인식하지 못함
- dataset이 생각보다 각도가 더 크게 돌아가 있는 경우가 존재

## 📍 **프로젝트 구현 내용**

---

- data cleansing을 통해 train set annotation의 박스 노이즈 제거
- 글자로 인식하는 배경, QR code를 train dataset에 추가하는 data preprocessing 적용
- CLAHE augmentation을 통해 글자 부분과 아닌 부분의 인식 강건화
- 기본 rotate10 augmentation에서 그 이상으로 각도를 변경하여 실험

## 💫 Final Model

---

- 5120_plus_background_epoch600_stepLr30_rotate20_clahe_seed1111
    - f1 score: 0.9666 → 0.9741(private)
    - epoch: 434
    - seed: 1111
    - augmentation: Base + CLAHE + rotate20
    - lr scheduler: StepLR(step_size: 30)
    - train dataset 자체에 배경 추가 preprocessing
- 5044_E800_Aug2_seed1111_tc
    - f1 score: 0.9697 → 0.9732(private)
    - epoch: 800
    - seed: 1111
    - augmentation: Base + CLAHE + rotate20
    - lr scheduler: StepLR(step_size: 30)
    - data cleansing 적용

## 📈 Result

---

- public : 14등

<img width="1112" alt="private score" src="https://github.com/boostcampaitech5/level2_cv_datacentric-cv-15/assets/113939970/2c6eefe3-4e26-4715-8723-4c1fd9b6db7e">

- private : 7등

<img width="1101" alt="public score" src="https://github.com/boostcampaitech5/level2_cv_datacentric-cv-15/assets/113939970/46bc3881-f0e9-4317-b378-a5c2827d1387">

## 🍀 Folder Structure

---

```

├── .github 
│   ├── ISSUE_TEMPLATE.md
│   └── PULL_REQUEST_TEMPLATE.md
├── .vscode 
│   └── settings.json
├── codebook : EDA, visualize등의 코드를 작성
│   ├── groupKfold.ipynb
│   ├── output_visualize.ipynb 
│   └── train_visualize.ipynb
├── .flake8
├── .gitignore
├── .gitmessage.txt
├── .pre-commit-config.yaml
├── README.md
├── base_config.json
├── config_train.py
├── poetry.toml
├── pyproject.toml
└── settings.sh
```

## 🔍 Reference 및 출처

---

- EAST dataset : https://github.com/SakuraRiven/EAST/tree/master
