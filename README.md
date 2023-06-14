![Untitled](https://github.com/boostcampaitech5/level2_cv_datacentric-cv-15/assets/113939970/7933a33c-c83a-4fc0-8049-f17744917e29)
## 📸 병원 진료비 영수증 OCR Task
![ocr](https://github.com/boostcampaitech5/level2_cv_datacentric-cv-15/assets/113939970/8dce5d17-64e1-48ae-997c-c915891b5d83)
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
|![logo1](https://github.com/boostcampaitech5/level2_cv_datacentric-cv-15/assets/113939970/4d28124d-57d8-42f0-a21e-23aa89e5d65d)|![logo2](https://github.com/boostcampaitech5/level2_cv_datacentric-cv-15/assets/113939970/dacec3d2-6e10-4066-924b-bb01f2729d37)|![logo3](https://github.com/boostcampaitech5/level2_cv_datacentric-cv-15/assets/113939970/f80da2a3-0509-4303-969d-71e6a3d1ee90)|![logo4](https://github.com/boostcampaitech5/level2_cv_datacentric-cv-15/assets/113939970/c9aec6cd-d9ce-429e-b554-ede08d7c8836)|![logo5](https://github.com/boostcampaitech5/level2_cv_datacentric-cv-15/assets/113939970/5f12c43f-0b6b-4f11-873e-1b81574d1354)|
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

<img width="1101" alt="public score" src="https://github.com/boostcampaitech5/level2_cv_datacentric-cv-15/assets/113939970/8381a215-8679-4806-91be-49c1eb25ba1a">

- private : 7등

<img width="1112" alt="private score" src="https://github.com/boostcampaitech5/level2_cv_datacentric-cv-15/assets/113939970/ef7101f2-a167-46ec-a661-203424b15e8d">

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
