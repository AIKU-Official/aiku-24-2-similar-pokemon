# 프로젝트명
📢 2024년 2/겨울학기 [AIKU](https://github.com/AIKU-Official) 활동으로 진행한 프로젝트입니다
🎉 2024년 2/겨울학기 AIKU Conference 열심히상 수상!

## 소개
AIKU에서는 매 기수마다 팀원들끼리 서로 닮은 연예인 또는 캐릭터 이름을 붙여주곤 하는데, 이걸 서비스로 만들어서 배포해두면 매 기수마다 보다 쉽게 별명을 지어줄 수 있지 않을까 생각하였습니다. 연예인과 캐릭터 중에 고민한 결과, 포켓몬스터 캐릭터가 구현 난이도가 좀 더 있을 것으로 생각되어 프로젝트의 목표를 **닮은 포켓몬 찾기 서비스 배포** 로 정하였습니다. 저희 프로젝트의 성공 기준은 아이쿠 회장 부회장과 비슷한 포켓몬을 찾아 학회원들에게 납득받기입니다.

## 방법론
- 사용자가 입력한 이미지와 가장 비슷한 포켓몬을 찾아주는, **Content Based Image Retrieval(CBIR)** 문제 해결
- 멀티모달/이미지 분야 연구에서 유사도 비교를 위해 가장 많이 사용되는 두 가지 metric인 **CLIP** 과 **DINOv2** 사용
  - input 이미지와 포켓몬 이미지 dataset의 **CLIP 및 DINOv2 임베딩 값을 비교**하여 가장 유사한 포켓몬을 retrieval
- **CLIP** : 이미지의 전반적인 구조, 색상 등의 **high level feature**에 집중
- **DINOv2** : 이미지의 디테일한 정보인 **low level feature**에 집중
    </br>=> 두 가지 벡터 검색의 결과를 가중치를 다르게 하여 **emsemble**, 벡터 검색 시 **FAISS library** 사용
- CLIP embedding 추출 과정에서,
  - ‘**두 이미지**의 **도메인**을 포켓몬스터 애니메이션으로 **맞추면** **성능이 향상될 것이다**’ 라는 가설에 따라
  - Stable Diffusion 모델을 포켓몬스터 이미지로 학습시킨 **sd-pokemon-diffusers 모델**을 Diffusion 기반의 **Plug-and-Play 모델**에 적용하여 **사용자의 이미지를 포켓몬스터 도메인으로 바꾸는 과정**을 CLIP score 비교 전에 실행
    
![image](https://github.com/user-attachments/assets/e683a31a-f580-4273-bf07-66f47f7051af)
<div align="center">Overall Architecture</div>

## 환경 설정
environment를 생성하고 아래 코드를 실행하여 dependency들을 설치

  ```
    conda create -n env_name python=3.9
    conda activate env_name
    pip install -r requirements.txt
  ```
    
## 사용 방법
### 1. dataset/celebs 폴더 안의 celeb benchmark 사용하는 경우
- **inference_celeb.sh** 파일 실행 혹은 아래 코드 실행
- 이때 **query_fp**는 **dataset/celebs** 폴더 안의 원하는 파일 경로로 수정
- 실행 시 output 폴더에 **retrieval_result.png**로 retrieval 결과가 저장됨
  
   ```
     export CUDA_VISIBLE_DEVICES=0
     python inference_celeb.py \
        --query_fp "/home/aikusrv04/pokemon/similar_pokemon/dataset/celebs/Paris Hilton.png" \
        --k 3
   ```
### 2. User가 직접 image 업로드하는 경우
- **dataset/images 폴더**에 원하는 input image 업로드
- **inference_user.sh** 파일 실행 혹은 아래 코드 실행
- 이때 **data_path**와 **query_fp**는 **dataset/images** 폴더 안의 원하는 파일 경로로 수정
- 실행 시 output 폴더에 **retrieval_result.png**로 retrieval 결과가 저장됨
  
   ```
    export CUDA_VISIBLE_DEVICES=0
    python pnp/preprocess.py \
        --data_path "/home/aikusrv04/pokemon/similar_pokemon/dataset/images/seungryong_kim.jpg" \
        --save_dir "/home/aikusrv04/pokemon/similar_pokemon/pnp/latents" \
        --start_index 0
    
    python pnp/pnp.py \
        --data_path "/home/aikusrv04/pokemon/similar_pokemon/dataset/images/seungryong_kim.jpg" \
        --save_dir "/home/aikusrv04/pokemon/similar_pokemon/dataset/pnp_images"\
        --start_index 0
    
    python inference_user.py \
        --query_fp "/home/aikusrv04/pokemon/similar_pokemon/dataset/images/seungryong_kim.jpg" \
        --k 3
   ```

## 예시 결과
![image](https://github.com/user-attachments/assets/da434bc0-4747-4f7f-9eac-1cdf3c10e1ab)
<div align="center">Beyonce 사진으로 retrieval 한 결과</div>

![image](https://github.com/user-attachments/assets/f3110386-e02a-4b8d-a7d0-96be0349b591)
<div align="center">Paris Hilton 사진으로 retrieval 한 결과</div>

## 팀원
- [정우성](정우성의 [github link](https://github.com/mung3477)): DINO & MODAL
- [김윤서](김윤서의 [github link](https://github.com/hiyseo)): DINO & MODAL
- [조윤지](조윤지의 [github link](https://github.com/robosun78)): CLIP & PNP & 코드 정리
- [정다현](정다현의 [github link](https://github.com/dhyun22)): CLIP & PNP & 코드 정리
- [성준영](성준영의 [github link](https://github.com/joonyeongs)): 상임고문
