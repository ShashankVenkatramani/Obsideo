//
//  PostTransactionViewController.swift
//  Matador
//
//  Created by Shanky(Prgm) on 4/6/19.
//  Copyright © 2019 Shashank Venkatramani. All rights reserved.
//

import UIKit
import LocalAuthentication
class PostTransactionViewController: UIViewController {
    var uid:String = ""
    @IBAction func sendPressed(_ sender: Any) {
        let context:LAContext = LAContext()
        
        if context.canEvaluatePolicy(.deviceOwnerAuthenticationWithBiometrics, error: nil){
            context.evaluatePolicy(LAPolicy.deviceOwnerAuthenticationWithBiometrics, localizedReason: "Authorize payment") { (pass, error) in
                if pass {
                    print("Accepted")
                } else {
                    print("Failed")
                }
            }
        } else {
            
        }
    }
    
    override func viewDidLoad() {
        super.viewDidLoad()
        
    }
}
