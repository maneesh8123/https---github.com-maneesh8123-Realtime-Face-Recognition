import 'package:camera/camera.dart';
import 'package:flutter/material.dart';
import 'package:google_mlkit_face_detection/google_mlkit_face_detection.dart';



class FaceDetectorPainter extends CustomPainter {
  FaceDetectorPainter(this.imageSize, this.results,  this.camDire2);
  final Size imageSize;
  final camDire2;
  late double scaleX, scaleY;
  dynamic results;
  late Face face;
  @override
  void paint(Canvas canvas, Size size) {
    final Paint paint = Paint()
      ..style = PaintingStyle.stroke
      ..strokeWidth = 2.5
      ..color = Colors.blue;
    for (String label in results.keys) {
      for (Face face in results[label]) {
        // face = results[label];
        scaleX = size.width / imageSize.width;
        scaleY = size.height / imageSize.height;
        canvas.drawRect(
         Rect.fromLTRB(
          camDire2 == CameraLensDirection.front
              ? (imageSize.width - face.boundingBox.right) * scaleX
              : face.boundingBox.left * scaleX,
          face.boundingBox.top * scaleY,
          camDire2 == CameraLensDirection.front
              ? (imageSize.width - face.boundingBox.left) * scaleX
              : face.boundingBox.right * scaleX,
          face.boundingBox.bottom * scaleY,
        ),
        // ignore: unnecessary_statements
        paint,
      );
        TextSpan span = TextSpan(
            style: TextStyle(
                color: Colors.red, fontSize: 15, fontWeight: FontWeight.bold),
            text: label);
        TextPainter textPainter = TextPainter(
            text: span,
            textAlign: TextAlign.left,
            textDirection: TextDirection.ltr);
        textPainter.layout();
        textPainter.paint(
          canvas,
          Offset(
              size.width - (60 + face.boundingBox.left.toDouble()) * scaleX,
              (face.boundingBox.top.toDouble() - 10) * scaleY),
        );
      }
    }
  }

  @override
  bool shouldRepaint(FaceDetectorPainter oldDelegate) {
    return oldDelegate.imageSize != imageSize || oldDelegate.results != results;
  }
}

RRect _scaleRect(
    {required Rect rect,
    required Size imageSize,
    required Size widgetSize,
    double? scaleX,
    double? scaleY}) {
  return RRect.fromLTRBR(
    (widgetSize.width - rect.left.toDouble() * scaleX!),
    rect.top.toDouble() * scaleY!,
    widgetSize.width - rect.right.toDouble() * scaleX,
    rect.bottom.toDouble() * scaleY,
    Radius.circular(0),
  );
}



// class FaceDetectorPainter extends CustomPainter {
//   FaceDetectorPainter(this.imageSize, this.results);
//   final Size imageSize;
//   //final List<DetectedObject> results;
//   //late double scaleX, scaleY;
//   dynamic results;
//   late Face face;
//   @override
//   void paint(Canvas canvas, Size size) {
//     double scaleX = size.width / imageSize.width;
//     double scaleY = size.height / imageSize.height;

//     final Paint paint = Paint()
//       ..style = PaintingStyle.stroke
//       ..strokeWidth = 2.0
//       ..color = Colors.pinkAccent;
//     for (String label in results.keys) {
//       for (Face face in results[label]) {
//         // face = results[label];
//         canvas.drawRect(
//         Rect.fromLTRB(
//           face.boundingBox.left * scaleX,
//           face.boundingBox.top * scaleY,
//           face.boundingBox.right * scaleX,
//           face.boundingBox.bottom * scaleY,
//         ),
//         paint,
//       );
//         TextSpan span =  TextSpan(
//             style: new TextStyle(
//                 color: Colors.red, fontSize: 18, fontWeight: FontWeight.bold),
//             text: label);
//         TextPainter textPainter =  TextPainter(
//             text: span,
//             textAlign: TextAlign.left,
//             textDirection: TextDirection.ltr);
//         textPainter.layout();
//         textPainter.paint(
//           canvas,
//           new Offset(
//               size.width - (60 + face.boundingBox.left) * scaleX,
//               (face.boundingBox.top-10) * scaleY),
//         );
//       }
//     }
//   }

//   @override
//   bool shouldRepaint(FaceDetectorPainter oldDelegate) {
//     return oldDelegate.imageSize != imageSize || oldDelegate.results != results;
//   }
// }

