import cv2
import mediapipe as mp
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import time

# Initialize Mediapipe
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands

# Kannada alphabet mapping (gesture -> Kannada letter)
KANNADA_ALPHABET = {
    'A': 'ಅ',   # a
    'B': 'ಬ',   # ba
    'C': 'ಚ',   # ca
    'D': 'ದ',   # da
    'E': 'ಎ',   # e
    'F': 'ಫ',   # pha
    'G': 'ಗ',   # ga
    'H': 'ಹ',   # ha
    'I': 'ಇ',   # i
    'J': 'ಜ',   # ja
    'K': 'ಕ',   # ka
    'L': 'ಲ',   # la
    'M': 'ಮ',   # ma
    'N': 'ನ',   # na
    'O': 'ಒ',   # o
    'P': 'ಪ',   # pa
    'Q': 'ಕ್ವ', # kva (approximation)
    'R': 'ರ',   # ra
    'S': 'ಸ',   # sa
    'T': 'ತ',   # ta
    'U': 'ಉ',   # u
    'V': 'ವ',   # va
    'W': 'ವ',   # va (same as V)
    'X': 'ಕ್ಷ', # ksha
    'Y': 'ಯ',   # ya
    'Z': 'ಜ',   # ja (similar sound)
}

# Vowels and consonants mapping
KANNADA_VOWELS = {
    'A': 'ಅ', 'AA': 'ಆ', 'I': 'ಇ', 'II': 'ಈ', 
    'U': 'ಉ', 'UU': 'ಊ', 'E': 'ಎ', 'EE': 'ಏ',
    'O': 'ಒ', 'OO': 'ಓ'
}

KANNADA_CONSONANTS = {
    'KA': 'ಕ', 'KHA': 'ಖ', 'GA': 'ಗ', 'GHA': 'ಘ',
    'CA': 'ಚ', 'CHA': 'ಛ', 'JA': 'ಜ', 'JHA': 'ಝ',
    'TA': 'ತ', 'THA': 'ಥ', 'DA': 'ದ', 'DHA': 'ಧ',
    'NA': 'ನ', 'PA': 'ಪ', 'PHA': 'ಫ', 'BA': 'ಬ',
    'BHA': 'ಭ', 'MA': 'ಮ', 'YA': 'ಯ', 'RA': 'ರ',
    'LA': 'ಲ', 'VA': 'ವ', 'SA': 'ಸ', 'SHA': 'ಶ',
    'HA': 'ಹ', 'LLA': 'ಳ'
}

class GestureRecognizer:
    def __init__(self):
        self.current_gesture = ""
        self.current_kannada_letter = ""
        self.accumulated_word = ""
        self.last_gesture_time = 0
        self.gesture_cooldown = 0.5  # seconds between recognitions
        self.confidence_threshold = 0.7
        
    def calculate_finger_states(self, hand_landmarks):
        """
        Calculate which fingers are extended.
        Returns: [thumb, index, middle, ring, pinky] - True if extended
        """
        landmarks = hand_landmarks.landmark
        
        # Get hand orientation (left or right)
        wrist = landmarks[0]
        thumb_tip = landmarks[4]
        is_right_hand = thumb_tip.x > wrist.x
        
        fingers = []
        
        # Thumb (special case - check horizontal distance)
        thumb_extended = False
        if is_right_hand:
            thumb_extended = landmarks[4].x < landmarks[3].x
        else:
            thumb_extended = landmarks[4].x > landmarks[3].x
        fingers.append(thumb_extended)
        
        # Other fingers (check if tip is above PIP joint)
        finger_tips = [8, 12, 16, 20]  # Index, Middle, Ring, Pinky
        finger_pips = [6, 10, 14, 18]
        
        for tip, pip in zip(finger_tips, finger_pips):
            fingers.append(landmarks[tip].y < landmarks[pip].y)
        
        return fingers
    
    def get_finger_curl(self, hand_landmarks):
        """Calculate how curled each finger is (0-1, 0=straight, 1=fully curled)"""
        landmarks = hand_landmarks.landmark
        
        curls = []
        
        # Thumb curl
        thumb_curl = abs(landmarks[4].y - landmarks[2].y)
        curls.append(thumb_curl)
        
        # Other fingers
        finger_tips = [8, 12, 16, 20]
        finger_bases = [5, 9, 13, 17]
        
        for tip, base in zip(finger_tips, finger_bases):
            curl = abs(landmarks[tip].y - landmarks[base].y)
            curls.append(curl)
        
        return curls
    
    def recognize_gesture(self, hand_landmarks):
        """
        Recognize hand gestures and map to Kannada alphabet.
        Returns recognized English letter (which maps to Kannada).
        """
        fingers = self.calculate_finger_states(hand_landmarks)
        thumb, index, middle, ring, pinky = fingers
        
        # Count extended fingers
        extended_count = sum(fingers)
        
        # Define gesture patterns matching ASL alphabet
        
        # A - Fist with thumb on side
        if extended_count <= 1 and not index and not middle and not ring and not pinky:
            return 'A'
        
        # B - All four fingers up, thumb tucked
        if not thumb and index and middle and ring and pinky:
            return 'B'
        
        # C - Curved hand
        if thumb and not index and not middle and not ring and not pinky:
            return 'C'
        
        # D - Index up, others down, thumb touches middle
        if not thumb and index and not middle and not ring and not pinky:
            return 'D'
        
        # E - All fingers curled down
        if extended_count == 0:
            return 'E'
        
        # F - Index and thumb make circle, others up
        if not thumb and not index and middle and ring and pinky:
            return 'F'
        
        # G - Index and thumb pointing sideways
        if thumb and index and not middle and not ring and not pinky:
            return 'G'
        
        # H - Index and middle extended sideways
        if not thumb and index and middle and not ring and not pinky:
            return 'H'
        
        # I - Pinky up only
        if not thumb and not index and not middle and not ring and pinky:
            return 'I'
        
        # K - Index and middle up, thumb between them
        if thumb and index and middle and not ring and not pinky:
            return 'K'
        
        # L - Thumb and index at 90 degrees
        if thumb and index and not middle and not ring and not pinky:
            return 'L'
        
        # M - Thumb under three fingers
        if not thumb and not index and not middle and not ring and not pinky:
            return 'M'
        
        # N - Thumb under two fingers
        if not thumb and not index and not middle and ring and pinky:
            return 'N'
        
        # O - All fingers form circle
        if not any(fingers):
            return 'O'
        
        # P - Index and middle down, others up
        if thumb and not index and not middle and ring and pinky:
            return 'P'
        
        # R - Index and middle crossed
        if not thumb and index and middle and not ring and not pinky:
            return 'R'
        
        # S - Fist with thumb in front
        if thumb and not index and not middle and not ring and not pinky:
            return 'S'
        
        # T - Thumb between index and middle
        if thumb and not index and not middle and not ring and not pinky:
            return 'T'
        
        # U - Index and middle up together
        if not thumb and index and middle and not ring and not pinky:
            return 'U'
        
        # V - Index and middle up apart (peace sign)
        if not thumb and index and middle and not ring and not pinky:
            return 'V'
        
        # W - Index, middle, ring up
        if not thumb and index and middle and ring and not pinky:
            return 'W'
        
        # Y - Thumb and pinky extended (hang loose)
        if thumb and not index and not middle and not ring and pinky:
            return 'Y'
        
        # Five fingers - all extended
        if extended_count == 5:
            return 'FIVE'
        
        return ""
    
    def add_letter(self):
        """Add current Kannada letter to word."""
        if self.current_kannada_letter and self.current_kannada_letter not in ['', ' ']:
            self.accumulated_word += self.current_kannada_letter
            print(f"Added: {self.current_gesture} ({self.current_kannada_letter}) | Word: {self.accumulated_word}")
            self.last_gesture_time = time.time()
    
    def clear_word(self):
        """Clear accumulated word."""
        self.accumulated_word = ""
        print("Word cleared")
    
    def backspace(self):
        """Remove last character."""
        if self.accumulated_word:
            self.accumulated_word = self.accumulated_word[:-1]
            print(f"Backspace | Word: {self.accumulated_word}")

def draw_kannada_text(image, text, position, font_size=50, color=(0, 255, 0)):
    """Draw Kannada text on image using PIL."""
    try:
        # Convert to PIL
        image_pil = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(image_pil)
        
        # Try to load Kannada font, fallback to default
        try:
            # Common Kannada font names - try multiple
            font_names = ["Nudi", "Tunga", "Kedage", "NotoSansKannada-Regular.ttf", 
                         "Lohit-Kannada", "kannada.ttf"]
            font = None
            for font_name in font_names:
                try:
                    font = ImageFont.truetype(font_name, font_size)
                    break
                except:
                    continue
            
            if font is None:
                # Use default font if no Kannada font found
                font = ImageFont.load_default()
        except:
            font = ImageFont.load_default()
        
        # Draw text (BGR to RGB conversion)
        draw.text(position, text, fill=color[::-1], font=font)
        
        # Convert back to OpenCV
        return cv2.cvtColor(np.array(image_pil), cv2.COLOR_RGB2BGR)
    except Exception as e:
        print(f"Error drawing text: {e}")
        # Fallback to OpenCV text
        cv2.putText(image, text, position, cv2.FONT_HERSHEY_SIMPLEX, 
                   1.5, color, 3, cv2.LINE_AA)
        return image

def main():
    recognizer = GestureRecognizer()
    
    # Open webcam
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Cannot open camera")
        return
    
    print("\n" + "="*50)
    print("   ಕನ್ನಡ Hand Gesture Recognition System")
    print("   Kannada Hastakshara Parichaya Paddhati")
    print("="*50)
    print("\nControls:")
    print("- Show hand gesture to recognize Kannada letter")
    print("- Press 'A' or 'ENTER' to add current letter to word")
    print("- Press 'SPACE' to add space")
    print("- Press 'BACKSPACE' to delete last letter")
    print("- Press 'C' to clear entire word")
    print("- Press 'Q' or 'ESC' to quit")
    print("="*50 + "\n")
    
    with mp_hands.Hands(
        model_complexity=0,
        min_detection_confidence=0.6,
        min_tracking_confidence=0.6,
        max_num_hands=1
    ) as hands:
        
        while cap.isOpened():
            success, image = cap.read()
            if not success:
                print("Ignoring empty camera frame.")
                continue
            
            # Flip and process image
            image = cv2.flip(image, 1)
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image_rgb.flags.writeable = False
            results = hands.process(image_rgb)
            image_rgb.flags.writeable = True
            image = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
            
            # Draw hand landmarks and recognize gesture
            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    # Draw landmarks
                    mp_drawing.draw_landmarks(
                        image,
                        hand_landmarks,
                        mp_hands.HAND_CONNECTIONS,
                        mp_drawing_styles.get_default_hand_landmarks_style(),
                        mp_drawing_styles.get_default_hand_connections_style()
                    )
                    
                    # Recognize gesture
                    current_time = time.time()
                    if current_time - recognizer.last_gesture_time > 0.3:
                        gesture = recognizer.recognize_gesture(hand_landmarks)
                        if gesture and gesture != 'FIVE':
                            recognizer.current_gesture = gesture
                            # Convert to Kannada
                            recognizer.current_kannada_letter = KANNADA_ALPHABET.get(gesture, gesture)
            else:
                recognizer.current_gesture = ""
                recognizer.current_kannada_letter = ""
            
            # Get image dimensions
            h, w = image.shape[:2]
            
            # Create semi-transparent overlay for info panel
            overlay = image.copy()
            cv2.rectangle(overlay, (10, 10), (w-10, 180), (30, 30, 30), -1)
            image = cv2.addWeighted(overlay, 0.6, image, 0.4, 0)
            
            # Display current detected letter (both English and Kannada)
            if recognizer.current_gesture:
                # English gesture name
                cv2.putText(image, f"Gesture: {recognizer.current_gesture}", 
                           (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 
                           0.8, (100, 255, 100), 2, cv2.LINE_AA)
                
                # Kannada letter - larger
                if recognizer.current_kannada_letter:
                    image = draw_kannada_text(image, recognizer.current_kannada_letter, 
                                             (30, 70), font_size=80, color=(0, 255, 0))
            else:
                cv2.putText(image, "Show hand gesture...", 
                           (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 
                           0.8, (150, 150, 150), 2, cv2.LINE_AA)
            
            # Display accumulated word
            if recognizer.accumulated_word:
                # Label
                cv2.putText(image, "Word:", 
                           (30, 120), cv2.FONT_HERSHEY_SIMPLEX, 
                           0.6, (255, 255, 100), 2, cv2.LINE_AA)
                # Kannada word
                image = draw_kannada_text(image, recognizer.accumulated_word, 
                                         (110, 95), font_size=60, color=(255, 255, 0))
            
            # Display instructions at bottom
            cv2.putText(image, "ENTER/A: Add | SPACE: Space | BACKSPACE: Delete | C: Clear | Q: Quit", 
                       (30, 160), cv2.FONT_HERSHEY_SIMPLEX, 
                       0.45, (200, 200, 200), 1, cv2.LINE_AA)
            
            # Show image
            cv2.imshow('Kannada Hand Gesture Recognition', image)
            
            # Handle keyboard input
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('q') or key == 27:  # q or ESC
                break
            elif key == ord('a') or key == 13:  # a or ENTER
                recognizer.add_letter()
            elif key == ord(' '):  # SPACE
                recognizer.accumulated_word += ' '
                print(f"Added space | Word: {recognizer.accumulated_word}")
            elif key == 8:  # BACKSPACE
                recognizer.backspace()
            elif key == ord('c'):  # c
                recognizer.clear_word()
    
    # Print final word
    if recognizer.accumulated_word:
        print(f"\nFinal Word: {recognizer.accumulated_word}")
    
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()