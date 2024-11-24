# -*- coding: utf-8 -*-
import pyttsx3

# Initialize the TTS engine
engine = pyttsx3.init()

# Set the voice (0 for male, 1 for female - adjust as needed)
voices = engine.getProperty('voices')
engine.setProperty('voice', voices[1].id)

# Set the speech rate (lower value for slower speech)
engine.setProperty('rate', 145)  # Adjust to a value below the default (around 200)

# Have the engine say something
engine.say("""Ánh sáng xanh lóe ra, cây kiếm thép nhằm thẳng vai trái hán tử trung niên phóng tới lẹ như chớp. Người thanh niên phóng kiếm chưa đến nơi đã rung cổ tay biến chiêu nhằm sang bên phải cổ đối phương. Đối thủ là một hán tử trung niên dựng kiếm lên đỡ nghe “choang” một tiếng. Tiếng ngân chưa tắt, mới chớp mắt mà hai bên đã trao đổi thêm ba chiêu. Vụt một cái, hán tử trung niên vung trường kiếm nhằm giữa mặt thanh niên chém xả xuống. Thanh niên né sang bên hữu tránh khỏi, rồi tiện tay trái phóng kiếm đâm chúc xuống chân địch nhân. Hai gã ra chiêu vừa mau lẹ vừa tàn độc, tưởng chừng như cuộc đấu ăn thua trí mạng.Trong luyện võ sảnh, có hai người ngồi phía đông. Bên trên là một đạo cô tuổi trạc bốn mươi đang mím môi, mặt mũi hầm hầm. Bên dưới là một ông già, tuổi ngoại năm mươi, giơ tay lên vuốt chòm râu dài ra chiều đắc ý. Hai người ngồi cách nhau hơn một trượng, sau lưng có đến hơn hai chục tên đồ đệ vừa nam vừa nữ. Phía tây, hơn mười người khách ngồi trên ghế lót đệm gấm xem cuộc đấu, nhìn không chớp mắt.Ngoài võ trường một lớn, một nhỏ giao đấu đã ngoài bảy mươi chiêu vẫn chưa phân hơn kém. Đột nhiên gã đứng tuổi đâm một kiếm quá mạnh, xiêu hẳn người đi, dường như sắp té nhào. Trong đám khách ngồi xem, một chàng trai mặc áo bào xanh thấy vậy phì cười. Y biết ngay thế là bất lịch sự, vội lấy tay che miệng.Ngay lúc ấy, gã trẻ tuổi phóng chưởng trái vào lưng gã đứng tuổi. Gã này tiến lên nửa bước tránh được, xoay mình lại, thanh trường kiếm tiện đà vòng tới, quát một tiếng “trúng”. Nhát kiếm đến nhanh như chớp, trúng vào đùi bên trái tên thanh niên. Chân gã khuỵu xuống, phải chống kiếm mới đứng vững lại được. Hắn còn toan đấu nữa, nhưng gã đứng tuổi đã tra kiếm vào vỏ, tươi cười hỏi: “Cảm ơn Chử sư đệ nhường nhịn. Sư đệ có đau không?”. Thanh niên kia mặt tái đi, mím môi đáp: “Đa tạ Cung sư huynh đã nương tay.”Lão già râu dài vẻ mặt hớn hở, mỉm cười nói: “Đông tông thắng ba trận rồi, vậy được ở lại Kiếm Hồ Cung năm năm nữa. Tân sư muội có thêm ý kiến gì chăng?” Vị đạo cô đứng tuổi nén giận đáp: “Tả sư huynh khéo dạy đồ đệ. Nhưng năm năm vừa qua chẳng hay sư huynh đã nghiên cứu Vô Lượng Ngọc Bích được chút gì chưa?” Lão già râu dài trừng mắt nhìn đạo cô, nghiêm mặt nói: “Sư muội quên qui củ bản phái rồi sao?” Đạo cô hừ một tiếng, rồi không nói gì nữa.Ông già họ Tả, tên gọi Tử Mục, là chưởng môn Đông tông của phái Vô Lượng Kiếm. Đạo cô kia họ Tân, đạo hiệu là Song Thanh, cầm đầu Tây tông Vô Lượng Kiếm.Nguyên phái Vô Lượng Kiếm chia làm Đông, Tây, Bắc ba tông. Nhưng Bắc tông suy sụp từ lâu, chỉ còn Đông và Tây là hai chi phái hưng thịnh và có lắm nhân tài. Phái Vô Lượng Kiếm sáng lập từ triều Hậu Đường đời Ngũ Đại, chưởng môn ở tại Kiếm Hồ Cung nước Nam Chiếu. Đến đời Tống Nhân Tông thì chia ra ba tông, cứ năm năm đồ đệ cả ba chi phái hội họp ở Kiếm Hồ Cung để đấu kiếm, bên nào thắng thì được ở cung Kiếm Hồ năm năm, đến năm thứ sáu lại mở cuộc đấu. Mỗi kỳ đấu năm trận, hễ thắng ba là được. Trong khoảng thời gian năm năm, phe thua dĩ nhiên là phải cố gắng tập dượt để rửa hận, mà phe thắng cũng chẳng dám chểnh mảng chút nào. Bốn mươi năm trước Bắc tông thua nặng, chưởng môn bực tức dẫn hết đệ tử qua Sơn Tây, không quay lại tỉ kiếm nữa. Hai mươi lăm năm qua, Đông và Tây tông thắng qua thua lại. Đông tông thắng được bốn lần, Tây tông được hai. Kỳ này, gã trung niên hán tử họ Cung đấu với gã trẻ tuổi họ Chử là trận thứ tư, Cung thắng. Thế là Đông tông thắng ba trận, trận thứ năm không cần phải đấu nữa.Các tân khách ngồi ở phía tây là những nhân sĩ võ lâm được mời đến chứng kiến. Họ đều là những nhân vật tiếng tăm lừng lẫy trong võ lâm ở Vân Nam, chỉ mình chàng thanh niên mặc áo xanh ngồi hàng cuối là hạng hậu bối vô danh. Chính chàng bật cười lúc gã họ Cung giả vờ lỡ trớn.Chàng thanh niên này theo võ sư Mã Ngũ Đức ở phủ Phổ Nhị tỉnh Vân Nam đến đây. Mã Ngũ Đức nguyên là một nhà buôn trà lớn, đã giàu có lại hiếu khách, có phong thái Mạnh Thường Quân, bao nhiêu khách giang hồ thất thế đến nhờ vả đều được tiếp đãi, vì vậy mà Mã quen biết nhiều, mặc dù võ công chỉ tầm thường. Lúc Mã Ngũ Đức giới thiệu chàng thanh niên áo xanh kia người họ Đoàn, Tả Tử Mục chả thèm để ý, vì tưởng chàng là đồ đệ Mã Ngũ Đức. Chính võ công Mã còn chưa vào đâu, huống chi là đồ đệ Mã nên Tả hà tiện cả đến câu khách sáo “ngưỡng mộ đã lâu”, chỉ khinh khỉnh chắp tay rồi dẫn vào ghế ngồi. Ngờ đâu anh chàng ngốc nghếch chẳng biết trời đất gì, thấy đồ đệ Tả Tử Mục giả vờ trượt chân dụ địch lại dám phì cười chế nhạo.Tả Tử Mục tươi cười nói: “Năm nay Tân sư muội đưa ra bốn tên đồ đệ kiếm thuật rất khá, trận thứ tư bọn ta thắng được chỉ nhờ may. Chử sư điệt nhỏ tuổi mà đã tới trình độ đó thì tiền đồ chưa biết đâu mà lường. Sau hạn năm năm này nữa hai bên Đông Tây chắc lại đổi chỗ, hà hà.” Nói xong y cười ha hả một hồi rồi quay sang chàng thanh niên họ Đoàn nói: “Vừa nãy tệ đồ đánh dứ đòn Trật Phác Bộ để thủ thắng, dường như Đoàn thế huynh không vừa ý. Vậy Đoàn thế huynh ra sân chỉ giáo cho y một vài miếng nên chăng? Người ta thường nói tướng giỏi không có quân hèn, Mã ngũ ca oai danh lừng lẫy khắp Vân Nam, môn đồ quyết không phải tay vừa.”Mã Ngũ Đức hơi đỏ mặt, vội đáp: “Đoàn huynh đệ đây không phải là đồ đệ ta đâu. Lão ca kiếm thuật mèo què đâu dám dạy ai. Tả hiền đệ chẳng nên buông lời giễu cợt. Nguyên Đoàn huynh đệ qua chơi tệ xá, biết ngu huynh sắp lên núi Vô Lượng, nói là núi Vô Lượng phong cảnh thanh u, liền theo tới đây để mở rộng nhãn quang mà thôi.”Tả Tử Mục nghĩ thầm:”Tưởng y là đồ đệ Mã Ngũ Đức thì mình còn nể mặt, không nỡ tuyệt tình, nếu chỉ là kẻ sơ giao thì hà tất mình phải e dè? Kẻ nào cả gan dám đến Kiếm Hồ Cung ngạo mạn mà mình để xuống núi yên lành thì còn chi là thể diện Tả Tử Mục này?”. Nghĩ vậy Tả liền cười nhạt, hỏi: “Xin Đoàn huynh cho biết đại hiệu là gì, môn hạ cao nhân nào?”Chàng thanh niên họ Đoàn cười đáp: “Tên tại hạ vẻn vẹn một chữ Dự, chưa hề học võ. Tại hạ phải cái tật hễ thấy người té nhào thì bất luận là té thật hay té giả vờ cũng phải phì cười chứ không nín được.” Tả Tử Mục thấy chàng ăn nói không có vẻ gì cung kính, bất giác tức giận hỏi: “Làm sao mà phải phì cười?” Đoàn Dự mở quạt giấy ra phe phẩy, tỉnh bơ đáp: “Người ta đứng hay ngồi thì có gì mà cười, nằm trên giường cũng chẳng ai đáng cười, chứ nằm lăn xuống đất thì phải cười chứ sao? Trừ trẻ con lên ba thì không kể”. Tả Tử Mục đã tức lên tận cổ, nhưng phải cố giữ vẻ trầm tĩnh, quay sang hỏi Mã Ngũ Đức: “Mã ngũ ca! Đoàn huynh đệ phải chăng là bạn thân với ngũ ca?”.""")

# Run the TTS engine
engine.runAndWait()
