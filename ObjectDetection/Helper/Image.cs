using OpenCvSharp;
using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Drawing;
using System.Linq;
using System.Runtime.InteropServices;
using System.Text;
using System.Threading.Tasks;

namespace ObjectDetection.Helper
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
