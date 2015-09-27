#ifndef ATLASDIALOG_H
#define ATLASDIALOG_H

#include <QDialog>

namespace Ui {
class AtlasDialog;
}
class AtlasDialog : public QDialog
{
    Q_OBJECT
public:
    explicit AtlasDialog(QWidget *parent);
    ~AtlasDialog();
    unsigned int atlas_index;
    std::string atlas_name;
    std::vector<unsigned int> roi_list;
    std::vector<std::string> roi_name;
signals:
    void need_update();
private slots:
    void on_add_atlas_clicked();

    void on_atlasListBox_currentIndexChanged(int index);

    void on_pushButton_clicked();

private:
    Ui::AtlasDialog *ui;
};

#endif // ATLASDIALOG_H
