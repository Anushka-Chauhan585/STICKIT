# STICKIT

An AI based interface which generates 3D DISNEY PIXAR stickers according to the image and prompt given.

While programming we began by using the HUGGING FACE reference and inculcated the stable diffusers model which was not able to provide us with the desired result. 

Therefore we moved towards the stable diffusers XL model, but the XL model required heavy processing and a lot of storage, which was not possible on VSCode, so we shifted to Google Collab. But, we still faced problems using SDXL model. 

Our second approach to our problem was that we divide the problem into two halves - in the first half, our program will read the image and generate an output, for which we will subsequently use an image to text model and for the second, our program will generate a prompt keeping in mind the characteristics and features of the original image, for which we would use a text to image model. 

However, we used the pix 2 pix model which was highly sustainable and provided us with the desired result. For the prompts, we tried inculcating OPEN AI, so that the user may prompt "Good Night" and the AI will be able to produce a person lying on the bed sleeping or tucked in bed etc. 

We were successfully able to achieve the desired results as shown below.


Here is a prototype of it.

![Screenshot (12)](https://github.com/user-attachments/assets/f21c0f1f-074d-4fa7-b16e-d6927f129d8a)

![Screenshot (14)](https://github.com/user-attachments/assets/581bd189-7436-4ddb-a399-1c6806c41158)

![Screenshot (11)](https://github.com/user-attachments/assets/375d3b96-50b0-4660-8063-29208e59f782)

![Screenshot (13)](https://github.com/user-attachments/assets/a994ba9a-56d7-40a1-9996-1c51f9fb98e4)

GRADIO INTERFACE - An interface designed specifically so that user can upload their image and recieve the desired output.

![Screenshot (15)](https://github.com/user-attachments/assets/e4186ffd-97c4-4779-84d7-9e41554d6c71)




