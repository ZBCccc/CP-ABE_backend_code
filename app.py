import base64

from flask import Flask, request, jsonify

import cpabe as cpabe
import policy

app = Flask(__name__)


# 定义全局变量
cpabe_pm, cpabe_mk, cpabe_rid, cpabe_sk, cpabe_pk, cpabe_pk_list, user_attributes, sender_attributes, access_structure, message, cpabe_ciphertext, policy_attributes = None, None, None, None, None, None, None, None, None, None, None, None

@app.route('/cpabe/setup', methods=['POST'])
def cpabe_setup():
    global cpabe_pm, cpabe_mk, cpabe_rid, cpabe_sk, cpabe_pk
    cpabe_pm, cpabe_mk, cpabe_rid, cpabe_sk, cpabe_pk = cpabe.setup()

    # 将 cpabe_pm 元祖拆分为单独的元素
    p, G1, G2, u, h, w, v = cpabe_pm
    # 将结果打包成字典并返回
    return jsonify({
        "pm": {
            "p": str(p),
            "G1": G1.to_json_serializable(),
            "G2": G2.to_json_serializable(),
            "u": u.to_json_serializable(),
            "h": h.to_json_serializable(),
            "w": w.to_json_serializable(),
            "v": v.to_json_serializable()
        },
        "mk": str(cpabe_mk),
        "rid": str(cpabe_rid),
        "sk": str(cpabe_sk),
        "pk": cpabe_pk.to_json_serializable()
    }), 200

@app.route('/cpabe/pubkg', methods=['POST'])
def cpabe_pubkg():
    global cpabe_pk_list, user_attributes
    data = request.get_json()
    str_user_attributes = data.get('user_attributes')
    user_attributes = [policy_attributes.index(attr) + 1 for attr in str_user_attributes if attr in policy_attributes]
    print(f"user_attributes: {user_attributes}")
    cpabe_pk_list = cpabe.pubKG(cpabe_pm, cpabe_mk, cpabe_pk, user_attributes)
    pk_1, pk_2, pk_3, pk_4 = cpabe_pk_list
    return jsonify({
        "pk_1": pk_1.to_json_serializable(),
        "pk_2": pk_2.to_json_serializable(),
        "pk_3": [pk.to_json_serializable() for pk in pk_3],
        "pk_4": [pk.to_json_serializable() for pk in pk_4]
    }), 200
    
@app.route('/cpabe/encrypt', methods=['POST'])
def cpabe_encrypt():
    global sender_attributes, access_structure, message, policy_attributes, cpabe_ciphertext

    data = request.get_json()
    message = data.get('message')
    bool_exp = data.get('policy')
    access_structure_msp, _ = policy.boolean_to_msp(bool_exp, False)
    policy_attributes = access_structure_msp.row_to_attrib
    access_structure = [[int(elem) for elem in row] for row in access_structure_msp.mat.tolist()]
    sender_attributes = [i + 1 for i in range(len(policy_attributes))]
    print(f"sender_attributes: {sender_attributes}")
    cpabe_ciphertext = cpabe.Encrypt(cpabe_pm, cpabe_mk, sender_attributes, access_structure, message)
    c_0, c_1, c_2, c_3, c_4 = cpabe_ciphertext
    c_0_base64 = base64.b64encode(c_0.toBytes()).decode('utf-8')

    return jsonify({
        "c_0": c_0_base64,
        "c_1": c_1.to_json_serializable(),
        "c_2": [c.to_json_serializable() for c in c_2],
        "c_3": [c.to_json_serializable() for c in c_3],
        "c_4": [c.to_json_serializable() for c in c_4]
    }), 200

@app.route('/cpabe/decrypt', methods=['POST'])
def cpabe_decrypt():
    if cpabe_ciphertext is None:
        return jsonify({"error": "Unable to assist in decryption, ciphertext is None"}), 400
    transform_ciphertext = cpabe.Transform(cpabe_pk_list, access_structure, sender_attributes, user_attributes, cpabe_ciphertext)
    if transform_ciphertext is None:
        return jsonify({"error": "Unable to assist in decryption, attributes do not match"}), 400
    decrypt_ciphertext = cpabe.Decrypt(cpabe_sk, transform_ciphertext, cpabe_ciphertext[0])
    if cpabe.fp12_from_str(message) == decrypt_ciphertext:
        return jsonify({
            "transform_ciphertext": base64.b64encode(transform_ciphertext.toBytes()).decode('utf-8'),
            "message": message
        }), 200
    else:
        return jsonify({"error": "Decryption failed"}), 400

if __name__ == '__main__':
    app.run(debug=True)
