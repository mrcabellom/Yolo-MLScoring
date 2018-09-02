using OpenCvSharp;
using System;


namespace ObjectDetectionLib.Helper
{
    public class Image
    {

        public static Mat ConvertFromBase64ToMat(string base64Image)
        {
            var byteImage = Convert.FromBase64String(base64Image);
            Mat src = Cv2.ImDecode(byteImage, ImreadModes.Color);
            return src;
        }
    }
}
