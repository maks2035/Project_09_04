#include <iostream>
#include <opencv2/core.hpp>
#include <opencv2/videoio.hpp>
#include <opencv2/opencv.hpp>

int main()
{
   cv::Mat image = cv::imread("D:/virandfpc/vir/Project_09_04/image.png");

   if (image.empty()) {
      std::cout << "Ошибка загрузки изображения" << std::endl;
      return -1;
   }

   cv::VideoWriter video("D:/virandfpc/vir/Project_09_04/output.mp4", cv::VideoWriter::fourcc('a', 'v', 'c', '1'), 30, cv::Size(image.cols / 2.5, image.rows / 2.5));
   if (!video.isOpened()) {
      std::cout << "Error: could not open video writer" << std::endl;
      return -1;
   }

   //cv::resize(image, image, cv::Size(), 0.5, 0.5);
   cv::VideoCapture cap("D:/virandfpc/vir/Project_09_04/ZUA.mp4");
   if (!cap.isOpened()) {
      std::cout << "Ошибка загрузки первого видео" << std::endl;
      return -1;
   }

   cv::CascadeClassifier face_cascade;
   if (!face_cascade.load(cv::samples::findFile("D:/virandfpc/haarcascades/haarcascade_frontalface_alt.xml"))) {
      std::cout << "ERROR" << std::endl;
      return -1;
   }

   cv::CascadeClassifier eye_cascade;
   if (!eye_cascade.load(cv::samples::findFile("D:/virandfpc/haarcascades/haarcascade_eye_tree_eyeglasses.xml"))) {
      std::cout << "ERROR" << std::endl;
      return -1;
   }

   cv::CascadeClassifier smile_cascade;
   if (!smile_cascade.load(cv::samples::findFile("D:/virandfpc/haarcascades/haarcascade_smile.xml"))) {
      std::cout << "ERROR" << std::endl;
      return -1;
   }

   while (true) {
      cv::Mat frame;
      cap >> frame;
      if (frame.empty()) break;

      cv::resize(frame, frame, cv::Size(), 0.5, 0.5);

      cv::Mat image_gray, gauss;
      cv::cvtColor(frame, image_gray, cv::COLOR_BGR2GRAY);
      cv::GaussianBlur(image_gray, gauss, cv::Size(3, 3), 0);

      std::vector<cv::Rect> faces;
      face_cascade.detectMultiScale(gauss, faces, 1.1);

      std::vector<cv::Rect> eyes;
      eye_cascade.detectMultiScale(gauss, eyes, 1.1);

      std::vector<cv::Rect> smiles;
      smile_cascade.detectMultiScale(gauss, smiles, 1.565, 30, 0, cv::Size(30, 30));

      for (const auto& face : faces) {
         cv::rectangle(frame, face, cv::Scalar(255, 0, 0), 2);
      }

      for (size_t j = 0; j < eyes.size(); j++) {
         cv::Point eye_center(eyes[j].x + eyes[j].width / 2, eyes[j].y + eyes[j].height / 2);
         int radius = cvRound((eyes[j].width + eyes[j].height) * 0.25);
         circle(frame, eye_center, radius, cv::Scalar(255, 0, 0), 1);
      }

      for (size_t j = 0; j < smiles.size(); j++) {
         cv::Point smile_center(smiles[j].x + smiles[j].width / 2, smiles[j].y + smiles[j].height / 2);
         int radius_x = cvRound(smiles[j].width * 0.25);
         int radius_y = cvRound(smiles[j].height * 0.25);

         cv::ellipse(frame, smile_center, cv::Size(radius_x, radius_y), 0, 0, 360, cv::Scalar(255, 0, 0), 1);
      }

      cv::imshow("faces detected", frame);
      cv::resize(frame, frame, cv::Size(image.cols / 2.5, image.rows / 2.5));
      video << frame;

      char c = (char)cv::waitKey(30);
      if (c == 27) break;
   }

   video.release();

   cap.release();
   cv::destroyAllWindows();
   return 0;
}
