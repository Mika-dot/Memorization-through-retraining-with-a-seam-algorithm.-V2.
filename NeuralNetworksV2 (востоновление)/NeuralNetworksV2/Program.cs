using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using NeuralNetworkV2;
using Energy;
using System.Drawing;
using System.Runtime.InteropServices;


namespace NeuralNetworksV2
{
    class Program
    {

        [DllImport("kernel32.dll", EntryPoint = "GetConsoleWindow", SetLastError = true)]
        private static extern IntPtr GetConsoleHandle();

        static void Main(string[] args)
        {
            PixelEnergy pixel = new PixelEnergy();

            string link = "2";
            int barrier = 20;


            Vector[] X = pixel.LowEnergyPixelsCoordinates("img\\" + link + ".png", barrier);

            Network network = new Network(new int[] { 2, 10, 15, 7, 3 });

            int selector = 0;
            int x = 150;

            network.InputLayers(); // ввод нейрона

            Bitmap bmp = new Bitmap("img\\" + link + ".png");

            for (int i = 0; i < X.GetLength(0); i++)
            {
                Vector OutputValues = network.Forward(X[i], selector);
                //Color gotColor1 = Color.FromArgb(255, 255, 0, 0);
                Color gotColor = Color.FromArgb(255, Convert.ToInt32(OutputValues[0] * 255), Convert.ToInt32(OutputValues[1] * 255), Convert.ToInt32(OutputValues[2] * 255));
                bmp.SetPixel(Convert.ToInt32(X[i][0] * bmp.Width), Convert.ToInt32(X[i][1] * bmp.Height), gotColor);
            }

            using (var graphics = Graphics.FromHwnd(GetConsoleHandle()))
            using (var image = bmp)
                graphics.DrawImage(image, 50 + x, 50, 250 + x, 200);


            Console.ReadKey();

        }
    }
}
