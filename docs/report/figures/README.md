# figures/

보고서에 들어가는 스크린샷을 여기에 둔다. 파일명은 `.tex`의 `\includegraphics`와
일치해야 한다.

| 파일명 | 내용 | 참조 위치 |
|---|---|---|
| `viewer_2d.png` | 2D 뷰어 결과 — 객체별 융합 거리 라벨(R/D/최종) | Part V 융합 (그림\ref{fig:viewer2d}) |
| `viewer_3d.png` | 3D 뷰어 결과 — 배경 포인트클라우드 + 객체 배치 | Part VIII 뷰어 (그림\ref{fig:viewer3d}) |

Overleaf에서는 이 폴더에 두 PNG를 업로드하면 `\graphicspath{{figures/}}`로 자동
인식된다. (이미지가 없으면 컴파일 에러가 나므로, 두 파일을 넣은 뒤 컴파일할 것.)
