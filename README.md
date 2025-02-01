# Introduce
Ứng dụng CLIP vào việc truy xuất image/caption dựa vào image hoặc prompt 

Dữ liệu gồm có bộ COCO2017 - 118k images - bộ Human Face 7000 images

Với bộ COCO2017, vì số lượng ảnh quá lớn nên sẽ embedding caption và dùng nó để truy vấn 

Với bộ Human Face, embedding image và dùng nó để truy vấn

Cả 2 bộ đều cho phép user sử dụng prompt và image để query 

## Getting Started
To get started with this project, follow the instructions below.

### How to run ? 

1. Clone the repository:

   git clone https://github.com/hoanglvuit/retrieve_images.git
   
2. Navigate to the project directory:

   cd retrieve_images
3. Install dependencies:
 
   pip install -r requirement.txt

4. You can run with:

  python manage.py runserver 

  Next, you have to open the link and see interface 

## Note: 
I only pushed the database of Human Face, if you want to try COCO, please wait some minutes


   

