
import org.opencv.core.*;
import org.opencv.dnn.*;
import org.opencv.highgui.HighGui;
import org.opencv.imgproc.Imgproc;
import org.opencv.videoio.VideoCapture;

import java.util.ArrayList;
import java.util.List;

public class RealTimeObjectDetection {

    // Class labels for MobileNet-SSD (21 classes)
    private static final String[] CLASS_NAMES = new String[]{
            "background", "aeroplane", "bicycle", "bird", "boat", "bottle",
            "bus", "car", "cat", "chair", "cow", "diningtable", "dog",
            "horse", "motorbike", "person", "pottedplant", "sheep", "sofa",
            "train", "tvmonitor"
    };

    public static void main(String[] args) {
        // args: <caffemodel> <prototxt>
        if (args.length < 2) {
            System.out.println("Usage: java RealTimeObjectDetection <caffemodel> <prototxt>");
            return;
        }

        // Load native library
        try {
            System.loadLibrary(Core.NATIVE_LIBRARY_NAME);
        } catch (UnsatisfiedLinkError e) {
            System.err.println("ERROR: Could not load OpenCV native library. Make sure -Djava.library.path points to native folder with opencv_java*.dll/.so/.dylib");
            e.printStackTrace();
            return;
        }

        String model = args[0];
        String config = args[1];

        // Create net
        Net net = Dnn.readNetFromCaffe(config, model);
        if (net.empty()) {
            System.err.println("Failed to load network. Check model and config paths.");
            return;
        }

        // Use CPU (default). If you have OpenCV built with CUDA and want GPU, set preferable backend/target accordingly.
        net.setPreferableBackend(Dnn.DNN_BACKEND_OPENCV);
        net.setPreferableTarget(Dnn.DNN_TARGET_CPU);

        // Video capture (default webcam)
        VideoCapture cap = new VideoCapture(0);
        if (!cap.isOpened()) {
            System.err.println("Cannot open camera (0). If you have multiple cameras, try changing index.");
            return;
        }

        Mat frame = new Mat();

        System.out.println("Starting real-time detection. Press ESC in the window to exit.");

        while (true) {
            if (!cap.read(frame) || frame.empty()) {
                System.err.println("No frame captured from camera, exiting.");
                break;
            }

            // Prepare blob (size 300x300 as used by MobileNet-SSD)
            Size inSize = new Size(300, 300);
            Mat blob = Dnn.blobFromImage(frame, 0.007843, inSize, new Scalar(127.5, 127.5, 127.5), false, false);

            net.setInput(blob);
            Mat detections = net.forward(); // shape: [1, 1, N, 7]

            // Parse detections
            Mat detectionsReshaped = detections.reshape(1, (int)detections.total() / 7);

            for (int i = 0; i < detectionsReshaped.rows(); i++) {
                double confidence = detectionsReshaped.get(i, 2)[0];
                if (confidence > 0.5) { // threshold
                    int classId = (int) detectionsReshaped.get(i, 1)[0];

                    // bounding box normalized
                    float xLeftBottom = (float) (detectionsReshaped.get(i, 3)[0] * frame.cols());
                    float yLeftBottom = (float) (detectionsReshaped.get(i, 4)[0] * frame.rows());
                    float xRightTop = (float) (detectionsReshaped.get(i, 5)[0] * frame.cols());
                    float yRightTop = (float) (detectionsReshaped.get(i, 6)[0] * frame.rows());

                    // draw
                    Point p1 = new Point(xLeftBottom, yLeftBottom);
                    Point p2 = new Point(xRightTop, yRightTop);

                    Imgproc.rectangle(frame, p1, p2, new Scalar(0, 255, 0), 2);

                    String label = "";
                    if (classId >= 0 && classId < CLASS_NAMES.length) {
                        label = CLASS_NAMES[classId];
                    } else {
                        label = "id:" + classId;
                    }
                    String caption = String.format("%s: %.2f", label, confidence);

                    int[] baseLine = new int[1];
                    Size labelSize = Imgproc.getTextSize(caption, Imgproc.FONT_HERSHEY_SIMPLEX, 0.5, 1, baseLine);
                    int top = (int)Math.max(yLeftBottom, labelSize.height);

                    Imgproc.rectangle(frame, new Point(xLeftBottom, top - labelSize.height - 8), new Point(xLeftBottom + labelSize.width, top + baseLine[0]), new Scalar(255, 255, 255), Imgproc.FILLED);
                    Imgproc.putText(frame, caption, new Point(xLeftBottom, top - 4), Imgproc.FONT_HERSHEY_SIMPLEX, 0.5, new Scalar(0,0,0));
                }
            }

            // Show
            HighGui.imshow("Real-Time Object Detection", frame);

            int key = HighGui.waitKey(1);
            if (key == 27) { // ESC
                break;
            }
        }

        cap.release();
        HighGui.destroyAllWindows();
    }
}

