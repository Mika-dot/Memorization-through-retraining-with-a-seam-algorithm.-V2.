using System;
using NeuralNetworkV2;
using Energy;
using System.Drawing;
using System.Runtime.InteropServices;
using System.Drawing.Imaging;
using System.IO;

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
            int barrier = 10; // 10 // 20 // 30


            Vector[] X = pixel.LowEnergyPixelsCoordinates("img\\" + link + ".png", barrier);
            Vector[] Y = pixel.LowEnergyPixels("img\\" + link + ".png", barrier);
            Directory.CreateDirectory("img\\" + link);


            Network network = new Network(new int[] { 2, 10, 15, 7, 3 });

            double alpha = 0.5;
            double eps = 1e-10;
            int selector = 0;
            int output = 1;
            int x = 150;

            int indextIR = 0; // записей
            int iteration = 0; // вывод при этирации
            double error; // ошибка эпохи

            //network.InputLayers(); // ввод нейрона

            do
            {
                error = 0; // обнуляем ошибку
                           // проходимся по всем элементам обучающего множества
                Random rand = new Random();

                for (int i = 0; i < X.Length; i++)
                {
                    int j = rand.Next(0, X.Length);
                    network.Forward(X[j], selector); // прямое распространение сигнала
                    network.Backward(Y[j], ref error); // обратное распространение ошибки
                    network.UpdateWeights(alpha); // обновление весовых коэффициентов
                    //Console.SetCursorPosition(0, 0);
                    //Console.WriteLine(i + " => " + X.Length);
                }

                iteration++;

                Console.SetCursorPosition(1, 0);
                Console.WriteLine("error: {0}", error); // выводим в консоль номер эпохи и величину ошибку
                Console.WriteLine((iteration * 100) / output + "%");
                Console.WriteLine(indextIR + " кол. этир.");

                if (iteration == output)
                {
                    network.OutputLayers(); // сохранение нейронов
                    Bitmap bmp = new Bitmap("img\\" + link + ".png");

                    for (int i = 0; i < X.GetLength(0); i++)
                    {
                        Vector OutputValues = network.Forward(X[i], selector);
                        //Color gotColor1 = Color.FromArgb(255, 255, 0, 0);
                        Color gotColor = Color.FromArgb(255, Convert.ToInt32(OutputValues[0] * 255), Convert.ToInt32(OutputValues[1] * 255), Convert.ToInt32(OutputValues[2] * 255));
                        bmp.SetPixel(Convert.ToInt32(X[i][0] * bmp.Width), Convert.ToInt32(X[i][1] * bmp.Height), gotColor);
                    }

                    bmp.Save("img\\" + link + "\\" + link + "-" + Convert.ToString(indextIR) + ".png", ImageFormat.Bmp);

                    using (var graphics = Graphics.FromHwnd(GetConsoleHandle()))
                    using (var image = bmp)
                        graphics.DrawImage(image, 50 + x, 50, 250 + x, 200);

                    indextIR++;
                    iteration = 0;
                }
            } while (error > eps);

            Console.ReadKey();

        }
    }
}
