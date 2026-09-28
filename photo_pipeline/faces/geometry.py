from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np


@dataclass(frozen=True, eq=False)
class Face:
    x: float
    y: float
    w: float
    h: float
    landmarks: np.ndarray = field(repr=False)
    score: float = 1.0

    @property
    def box(self) -> tuple[int, int, int, int]:
        return int(round(self.x)), int(round(self.y)), int(round(self.w)), int(round(self.h))

    @property
    def center(self) -> np.ndarray:
        return np.array([self.x + self.w / 2.0, self.y + self.h / 2.0], dtype=np.float32)


@dataclass(frozen=True, eq=False)
class FaceGeometry:
    eye_left: np.ndarray
    eye_right: np.ndarray
    nose: np.ndarray
    mouth_left: np.ndarray
    mouth_right: np.ndarray
    box: tuple[float, float, float, float]
    across: np.ndarray
    down: np.ndarray
    angle: float
    unit: float
    eye_distance: float
    is_profile: bool

    @classmethod
    def from_face(cls, face: Face, profile_eye_ratio: float = 0.25) -> "FaceGeometry":
        points = np.asarray(face.landmarks, dtype=np.float32).reshape(5, 2)
        eyes = sorted((points[0], points[1]), key=lambda p: float(p[0]))
        mouth = sorted((points[3], points[4]), key=lambda p: float(p[0]))
        nose = points[2]
        vector = eyes[1] - eyes[0]
        eye_distance = float(np.hypot(vector[0], vector[1]))
        if eye_distance > 1e-3:
            across = (vector / eye_distance).astype(np.float32)
        else:
            across = np.array([1.0, 0.0], dtype=np.float32)
        down = np.array([-across[1], across[0]], dtype=np.float32)
        angle = math.degrees(math.atan2(float(across[1]), float(across[0])))
        position = float(np.dot(nose - eyes[0], across)) / max(eye_distance, 1e-3)
        is_profile = eye_distance < profile_eye_ratio * face.w or not (0.05 <= position <= 0.95)
        unit = max(eye_distance, 0.38 * face.w)
        return cls(
            eye_left=eyes[0],
            eye_right=eyes[1],
            nose=nose,
            mouth_left=mouth[0],
            mouth_right=mouth[1],
            box=(face.x, face.y, face.w, face.h),
            across=across,
            down=down,
            angle=angle,
            unit=float(unit),
            eye_distance=eye_distance,
            is_profile=bool(is_profile),
        )

    @property
    def eye_mid(self) -> np.ndarray:
        return (self.eye_left + self.eye_right) / 2.0

    @property
    def mouth_mid(self) -> np.ndarray:
        return (self.mouth_left + self.mouth_right) / 2.0

    @property
    def face_width(self) -> float:
        return float(self.box[2])

    def shifted(self, dx: float, dy: float) -> "FaceGeometry":
        offset = np.array([dx, dy], dtype=np.float32)
        x, y, w, h = self.box
        return FaceGeometry(
            eye_left=self.eye_left + offset,
            eye_right=self.eye_right + offset,
            nose=self.nose + offset,
            mouth_left=self.mouth_left + offset,
            mouth_right=self.mouth_right + offset,
            box=(x + dx, y + dy, w, h),
            across=self.across,
            down=self.down,
            angle=self.angle,
            unit=self.unit,
            eye_distance=self.eye_distance,
            is_profile=self.is_profile,
        )
