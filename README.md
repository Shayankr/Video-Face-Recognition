# VideoFaceRecog: Face Recognition from Videos

VideoFaceRecog is a Python-based deep learning project that focuses on face recognition from videos. It leverages deep neural networks to detect, track, and recognize faces in video sequences, enabling applications such as surveillance, identity verification, and access control. The project also provides deployment options using AWS cloud services for scalable and reliable application hosting.

## Features

- <b>Face detection and tracking:</b> The project utilizes advanced computer vision techniques to detect and track faces in videos.
- <b>Face recognition:</b> It employs deep learning models to recognize and match faces against a database of known identities.
- <b>Real-time processing:</b> The system is optimized for real-time video processing, ensuring efficient and responsive face recognition.
- <b>Configurable parameters:</b> Various parameters can be adjusted, such as confidence thresholds, frame rate, and face recognition models.
- <b>Easy-to-use API:</b> Provides a simple API for integrating the face recognition functionality into your own applications.

## Installation

1. Clone the repository:

```
git clone https://github.com/Shayankr/Video-Face-Recognition.git
```

2. Install the required dependencies:
```
pip install -r requirements.txt
```

## Usage

1. Prepare your video files and ensure they are accessible from the project directory.
2. Modify the configuration parameters in the <b>config.py</b> file to suit your requirements.
3. Run the main script to start the face recognition system:
```
python main.py
```

## Deployment on AWS Cloud

To deploy the VideoFaceRecog application on AWS cloud for public use, follow these steps:

1. Set up an AWS account and configure the necessary permissions and security credentials.
2. Choose an appropriate AWS service for hosting your application, such as AWS Lambda, Amazon EC2, or Amazon ECS with Docker or use podman (whichever is suitable/preferred/available).
3. Package your application and any required dependencies into a deployment package.
4. Set up the necessary AWS resources (e.g., VPC, subnets, security groups) to support your deployment.
5. Deploy the application to the chosen AWS service, following the relevant documentation and best practices for that service.
6. Configure the necessary networking, load balancing, and scalability features as required by your application.
7. Test and monitor the deployed application to ensure it is functioning correctly.

Refer to the AWS documentation and resources for detailed instructions on deploying applications using various AWS services.

## Contributing

Contributions to VideoFaceRecog are welcome! If you find any issues or have suggestions for improvements, please open an issue or submit a pull request.

## License

This project is licensed under the MIT License.

## Acknowledgements

The development of VideoFaceRecog is made possible by the following open-source libraries and resources:

- OpenCV: <a href="https://opencv.org/" target="_blank">https://opencv.org/</a>
- PyTorch: <a href="https://pytorch.org/" target="_blank">https://pytorch.org/</a>
- dlib: <a href="http://dlib.net/" target="_blank">http://dlib.net//</a>
- Pre-trained face recognition models (e.g., ResNet, ArcFace): [!pip install arcface](https://pypi.org/project/arcface/)

Please refer to the documentation and licenses of these libraries for more information.

## Contact

For any questions or inquiries, please contact shayankumar765@gmail.com, [text-me-on-linkedin](https://www.linkedin.com/in/shayan-kumar-187164a6/).

